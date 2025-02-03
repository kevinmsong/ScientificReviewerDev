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
            agent = ChatOpenAI(temperature=0.1, openai_api_key=st.secrets["openai_api_key"], model="gpt-4o")
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
        -2: "Be extremely thorough and critical. Focus on weaknesses and flaws.",
        -1: "Maintain high standards. Carefully identify both strengths and weaknesses.",
        0: "Provide balanced review of strengths and weaknesses.",
        1: "Emphasize positive aspects while noting necessary improvements.",
        2: "Take an encouraging approach while noting critical issues."
    }
    
    scales = {
        "Paper Score (-2 to 2)": "Score from -2 (worst) to 2 (best)",
        "Star Rating (1-5)": "Rate from 1 to 5 stars",
        "NIH Scale (1-9)": "Score from 1 (exceptional) to 9 (poor)"
    }
    
    return f"{prompt}\n\nReview Style: {styles[style]}\n\nRating: {scales[rating_scale]}"

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

def process_review_memoryless(content: str, agents: List[Union[ChatOpenAI, Any]], expertises: List[Dict], custom_prompts: List[str]) -> Dict[str, Any]:
    review_results = []
    scores = []
    
    for i, (agent, expertise, prompt) in enumerate(zip(agents, expertises, custom_prompts)):
        st.write(f"Processing review from {expertise['name']}...")
        try:
            review_text = process_chunk_memoryless(content, agent, expertise['name'], prompt, expertise['model'])
            review_results.append({
                "expertise": expertise,
                "review": review_text,
                "success": True
            })
            
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
                "success": False
            })
    
    if scores:
        st.subheader("Score Summary")
        avg_score = sum(scores) / len(scores)
        st.metric("Average Score", f"{avg_score:.2f}")
        description = get_score_description(st.session_state.get('rating_scale', 'Paper Score (-2 to 2)'), avg_score)
        st.write(f"Description: {description}")
    
    return {"reviews": review_results, "success": True}

def scientific_review_page():
    st.set_page_config(page_title="Scientific Reviewer", layout="wide")
    st.header("Scientific Review System")
    st.caption("v2.2.0 - Memoryless AI Mode")
    
    col1, col2 = st.columns([2,1])
    with col1:
        try:
            rating_scale = st.radio(
            "Rating Scale",
            ["Paper Score (-2 to 2)", "Star Rating (1-5)", "NIH Scale (1-9)"],
            help="Paper: -2 (worst) to 2 (best)\nStar: 1-5 stars\nNIH: 1 (best) to 9 (worst)"
        )
            st.session_state['rating_scale'] = rating_scale
        except Exception as e:
            st.error("Error setting up rating scale")
            logging.error(f"Rating scale error: {str(e)}")
            return

    review_type = st.selectbox("Select Review Type", ["Paper", "Grant", "Poster"])
    num_reviewers = st.number_input("Number of Reviewers", 1, 10, 2)
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
        process_review_memoryless(content, agents, expertises, custom_prompts)

if __name__ == "__main__":
    scientific_review_page()
