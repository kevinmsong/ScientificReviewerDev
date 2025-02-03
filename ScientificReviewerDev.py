import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import fitz
import io
from PIL import Image
import base64
from typing import List, Dict, Any, Tuple, Union
import tiktoken
import google.generativeai as genai
import re

logging.basicConfig(level=logging.INFO)

def get_score_description(rating_scale: str, score: float) -> str:
    descriptions = {
        "Paper Score (-2 to 2)": {
            -2: "Fundamentally Flawed", -1: "Significant Concerns",
            0: "Average", 1: "Strong Potential", 2: "Exceptional"
        },
        "Star Rating (1-5)": {
            1: "Poor", 2: "Below Average", 3: "Average",
            4: "Good", 5: "Excellent"
        },
        "NIH Scale (1-9)": {
            1: "Exceptional", 3: "Highly Meritorious",
            5: "Competitive", 7: "Marginal", 9: "Poor"
        }
    }
    
    scale_mapping = descriptions.get(rating_scale, {})
    if rating_scale == "Paper Score (-2 to 2)":
        rounded_score = round(score)
    elif rating_scale == "Star Rating (1-5)":
        rounded_score = min(max(round(score), 1), 5)
    else:
        scale_values = [1, 3, 5, 7, 9]
        rounded_score = min(scale_values, key=lambda x: abs(x - score))
    
    return scale_mapping.get(rounded_score, f"Score {score}")

def create_memoryless_agents(expertises: List[Dict], include_moderator: bool = False) -> List[Union[ChatOpenAI, Any]]:
    agents = []
    for expertise in expertises:
        if expertise["model"] == "GPT-4o":
            # Adjust temperature based on critique style
            temp = 0.1 + (expertise.get("style", 0) * 0.1)
            temp = max(0.1, min(0.7, temp))
            agent = ChatOpenAI(temperature=temp, openai_api_key=st.secrets["openai_api_key"], model="gpt-4o")
        else:
            genai.configure(api_key=st.secrets["gemini_api_key"])
            agent = genai.GenerativeModel("gemini-2.0-flash-exp")
        agents.append(agent)
    
    if include_moderator and len(expertises) > 1:
        moderator_agent = ChatOpenAI(temperature=0.1, openai_api_key=st.secrets["openai_api_key"], model="gpt-4o")
        agents.append(moderator_agent)
    
    return agents

def adjust_prompt_style(prompt: str, style: int, rating_scale: str) -> str:
    styles = {
        -2: "Be extremely thorough and critical. Focus primarily on identifying weaknesses, flaws, and areas needing major improvement. Be direct and emphasize problems that must be addressed.",
        -1: "Maintain high academic standards. Carefully identify both strengths and weaknesses, with particular attention to methodological and technical issues. Be constructively critical.",
        0: "Provide a balanced evaluation of strengths and weaknesses. Give equal attention to positive aspects and areas for improvement.",
        1: "Focus primarily on strengths while noting necessary improvements. Frame critiques constructively and emphasize potential.",
        2: "Take an encouraging and supportive approach. Highlight strengths and frame weaknesses as opportunities for enhancement. Maintain scientific rigor while being constructive."
    }
    
    scales = {
        "Paper Score (-2 to 2)": "Score from -2 (worst) to 2 (best)",
        "Star Rating (1-5)": "Rate from 1 to 5 stars",
        "NIH Scale (1-9)": "Score from 1 (exceptional) to 9 (poor)"
    }
    
    style_header = f"\n\nReview Style Guidelines:\n{styles[style]}"
    rating_header = f"\nRating Scale: {scales[rating_scale]}"
    
    return f"{prompt}{style_header}{rating_header}"

def process_chunk_memoryless(chunk: str, agent: Union[ChatOpenAI, Any], expertise: str, prompt: str, model_type: str) -> str:
    chunk_prompt = f"""Reviewing content:
{prompt}

Content:
{chunk}"""
    
    try:
        if model_type == "GPT-4o":
            response = agent.invoke([HumanMessage(content=chunk_prompt)])
            return response.content
        else:
            response = agent.generate_content(chunk_prompt)
            return response.text
    except Exception as e:
        logging.error(f"Error processing chunk for {expertise}: {str(e)}")
        return f"[Error processing content]"

def extract_pdf_content(pdf_file) -> Tuple[str, List[Image.Image]]:
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text_content = ""
    images = []
    
    for page in pdf_document:
        text_content += page.get_text()
        for img in page.get_images():
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    
    return text_content, images

def get_default_prompt(review_type: str, expertise: str) -> str:
    prompts = {
        "Paper": f"""As an expert in {expertise}, review this paper considering:
1. Scientific Merit
2. Methodology
3. Data Analysis
4. Clarity
5. Impact""",
        "Grant": f"""As an expert in {expertise}, evaluate this grant proposal considering:
1. Innovation
2. Methodology
3. Feasibility
4. Budget
5. Impact""",
        "Poster": f"""As an expert in {expertise}, review this poster considering:
1. Visual Appeal
2. Content
3. Methodology
4. Results
5. Impact"""
    }
    return prompts.get(review_type, f"Review this {review_type.lower()}")

def generate_review_summary(all_reviews: List[List[Dict]], scores: List[float]) -> str:
    summary = "# Scientific Review Summary\n\n"
    
    if scores:
        avg_score = sum(scores) / len(scores)
        summary += f"## Overall Score: {avg_score:.2f}\n\n"
        description = get_score_description(st.session_state.get('rating_scale', 'Paper Score (-2 to 2)'), avg_score)
        summary += f"Assessment: {description}\n\n"
    
    for iteration_idx, iteration in enumerate(all_reviews, 1):
        summary += f"## Iteration {iteration_idx}\n\n"
        for review in iteration:
            if review["success"]:
                summary += f"### Review by {review['expertise']['name']}\n"
                summary += f"Model: {review['expertise']['model']}\n"
                summary += f"Style: {review['expertise']['style']}\n\n"
                summary += f"{review['review']}\n\n"
    
    return summary

def process_review_memoryless(content: str, agents: List[Union[ChatOpenAI, Any]], expertises: List[Dict], 
                            custom_prompts: List[str], num_iterations: int = 1) -> Dict[str, Any]:
    review_results = []
    scores = []
    all_reviews = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    tabs = st.tabs([f"Iteration {i+1}" for i in range(num_iterations)])
    
    total_steps = num_iterations * len(agents)
    current_step = 0
    
    for iteration in range(num_iterations):
        iteration_reviews = []
        with tabs[iteration]:
            st.write(f"Starting iteration {iteration + 1}")
            for i, (agent, expertise, prompt) in enumerate(zip(agents, expertises, custom_prompts)):
                status_text.text(f"Processing review from {expertise['name']} (Iteration {iteration + 1})")
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                try:
                    review_text = process_chunk_memoryless(content, agent, expertise['name'], prompt, expertise['model'])
                    review_data = {
                        "expertise": expertise,
                        "review": review_text,
                        "success": True,
                        "iteration": iteration + 1
                    }
                    review_results.append(review_data)
                    iteration_reviews.append(review_data)
                    
                    score_matches = re.findall(r'score[:\s]*(-?\d+\.?\d*)', review_text.lower())
                    if score_matches:
                        try:
                            scores.append(float(score_matches[0]))
                        except ValueError:
                            pass
                            
                    with st.expander(f"Review by {expertise['name']} ({expertise['model']})", expanded=True):
                        st.markdown(review_text)
                        col1, col2 = st.columns([1,2])
                        with col1:
                            st.caption(f"Critique Style: {expertise['style']}")
                            
                except Exception as e:
                    logging.error(f"Error processing agent {expertise['name']}: {str(e)}")
                    review_results.append({
                        "expertise": expertise,
                        "review": f"Error: {str(e)}",
                        "success": False,
                        "iteration": iteration + 1
                    })
        
        all_reviews.append(iteration_reviews)

    status_text.empty()
    progress_bar.empty()
    
    if review_results:
        summary_md = generate_review_summary(all_reviews, scores)
        st.download_button(
            label="Download Review Summary",
            data=summary_md,
            file_name="review_summary.md",
            mime="text/markdown",
        )
    
    return {"reviews": review_results, "success": True}

def scientific_review_page():
    st.set_page_config(page_title="Scientific Reviewer", layout="wide")
    st.header("Scientific Review System")
    st.caption("v2.2.0 - Memoryless AI Mode")
    
    col1, col2 = st.columns([2,1])
    with col1:
        rating_scale = st.radio(
            "Rating Scale",
            ["Paper Score (-2 to 2)", "Star Rating (1-5)", "NIH Scale (1-9)"],
            help="Paper: -2 (worst) to 2 (best)\nStar: 1-5 stars\nNIH: 1 (best) to 9 (worst)"
        )
        st.session_state['rating_scale'] = rating_scale
    
    review_type = st.selectbox("Select Review Type", ["Paper", "Grant", "Poster"])
    col3, col4 = st.columns(2)
    with col3:
        num_reviewers = st.number_input("Number of Reviewers", 1, 10, 2)
    with col4:
        num_iterations = st.number_input("Number of Discussion Iterations", 1, 5, 1, help="Number of review rounds")
    
    use_moderator = st.checkbox("Include Moderator", value=True) if num_reviewers > 1 else False
    
    expertises = []
    custom_prompts = []
    
    with st.expander("Configure Reviewers"):
        for i in range(num_reviewers):
            st.subheader(f"Reviewer {i+1}")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                expertise = st.text_input(f"Expertise", value=f"Expert {i+1}", key=f"expertise_{i}")
                model_type = st.selectbox("Model", ["GPT-4o", "Gemini 2.0 Flash"], key=f"model_{i}")
            with col2:
                prompt = st.text_area("Review Guidelines", value=get_default_prompt(review_type, expertise), key=f"prompt_{i}")
            with col3:
                critique_style = st.slider(
                    "Critique Style",
                    min_value=-2,
                    max_value=2,
                    value=-1,
                    help="-2: Extremely harsh, 2: Extremely lenient",
                    key=f"style_{i}"
                )
            
            expertises.append({
                "name": expertise,
                "model": model_type,
                "style": critique_style
            })
            custom_prompts.append(adjust_prompt_style(prompt, critique_style, rating_scale))
    
    uploaded_file = st.file_uploader(f"Upload {review_type} (PDF)", type=["pdf"])
    
    if uploaded_file and st.button("Start Review"):
        content, _ = extract_pdf_content(uploaded_file)
        agents = create_memoryless_agents(expertises, use_moderator)
        process_review_memoryless(content, agents, expertises, custom_prompts, num_iterations)

if __name__ == "__main__":
    scientific_review_page()
