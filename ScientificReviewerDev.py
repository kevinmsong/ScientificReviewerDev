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
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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
            temp = 0.1 + (expertise.get("style", 0) * 0.1)
            temp = max(0.1, min(0.7, temp))
            agent = ChatOpenAI(temperature=temp, openai_api_key=st.secrets["openai_api_key"], model="gpt-4o")
        else:
            genai.configure(api_key=st.secrets["gemini_api_key"])
            agent = genai.GenerativeModel("gemini-2.0-flash-exp")
        agents.append(agent)
    
    if include_moderator and len(expertises) > 1:
        # Force moderator to use GPT-4o
        expertises.append({"model": "GPT-4o", "name": "Moderator"})
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

def process_review_memoryless(content: str, agents: List[Union[ChatOpenAI, Any]], expertises: List[Dict], 
                         custom_prompts: List[str], *, num_iterations: int = 1) -> Dict[str, Any]:
    review_results = []
    scores = []
    all_reviews = []
    previous_iteration = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    tabs = st.tabs([f"Iteration {i+1}" for i in range(num_iterations)] + ["Final Analysis"])
    
    total_steps = num_iterations * len(agents) * 2
    current_step = 0
    
    for iteration in range(num_iterations):
        iteration_reviews = []
        with tabs[iteration]:
            st.write(f"Starting iteration {iteration + 1}")
            
            # Reviews phase
            for i, (agent, expertise, base_prompt) in enumerate(zip(agents[:-1] if len(agents) > len(expertises) else agents, expertises, custom_prompts)):
                status_text.text(f"Processing review from {expertise['name']} (Iteration {iteration + 1})")
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                try:
                    if iteration > 0:
                        prompt = get_iteration_prompt(expertise['name'], iteration + 1, previous_iteration, "paper", st.session_state['rating_scale'])
                        review_text = process_chunk_memoryless(content, agent, expertise['name'], f"{base_prompt}\n\n{prompt}", expertise['model'])
                    else:
                        review_text = process_chunk_memoryless(content, agent, expertise['name'], base_prompt, expertise['model'])
                    
                    review_data = {
                        "expertise": expertise,
                        "review": review_text,
                        "success": True,
                        "iteration": iteration + 1
                    }
                    iteration_reviews.append(review_data)
                    
                    with st.expander(f"Review by {expertise['name']}", expanded=True):
                        st.markdown(review_text)
                        st.caption(f"Critique Style: {expertise['style']}")
                
                except Exception as e:
                    st.error(f"Error processing review: {str(e)}")
                    iteration_reviews.append({
                        "expertise": expertise,
                        "review": f"Error: {str(e)}",
                        "success": False,
                        "iteration": iteration + 1
                    })
            
            # Expert dialogue phase
            if len(expertises) > 1:
                st.subheader("Expert Dialogue")
                for i, review in enumerate(iteration_reviews):
                    if review["success"]:
                        status_text.text(f"Processing dialogue for {review['expertise']['name']} (Iteration {iteration + 1})")
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                        
                        dialogue_prompt = get_expert_dialogue_prompt(iteration_reviews, review['expertise']['name'], st.session_state['rating_scale'])
                        try:
                            if review['expertise']['model'] == "GPT-4o":
                                agent = agents[i]
                                response = agent.invoke([HumanMessage(content=dialogue_prompt)])
                                dialogue = response.content
                            else:
                                agent = agents[i]
                                response = agent.generate_content(dialogue_prompt)
                                dialogue = response.text
                            
                            review['dialogue'] = dialogue
                            with st.expander(f"Response from {review['expertise']['name']}", expanded=True):
                                st.markdown(dialogue)
                                
                        except Exception as e:
                            st.error(f"Error in expert dialogue: {str(e)}")
            
            all_reviews.append(iteration_reviews)
            previous_iteration = iteration_reviews.copy()

            # Extract scores from both reviews and dialogues
            for review in iteration_reviews:
                if review["success"]:
                    score_matches = re.findall(r'score[:\s]*(-?\d+\.?\d*)', review['review'].lower())
                    score_matches.extend(re.findall(r'score[:\s]*(-?\d+\.?\d*)', review.get('dialogue', '').lower()))
                    if score_matches:
                        try:
                            scores.append(float(score_matches[-1]))
                        except ValueError:
                            pass

    # Final Analysis Tab
    with tabs[-1]:
        st.subheader("Review Summary")
        if scores:
            avg_score = sum(scores) / len(scores)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Score", f"{avg_score:.2f}")
            with col2:
                st.write("Assessment:", get_score_description(st.session_state.get('rating_scale', 'Paper Score (-2 to 2)'), avg_score))
        
        # Moderator Analysis (if enabled)
        if len(agents) > len(expertises):
            st.subheader("Moderator Analysis")
            status_text.text("Generating moderator analysis...")
            try:
                moderator_agent = agents[-1]
                all_dialogue = []
                for iteration in all_reviews:
                    for review in iteration:
                        if review["success"]:
                            all_dialogue.append(f"{review['expertise']['name']} Review:\n{review['review']}")
                            if review.get('dialogue'):
                                all_dialogue.append(f"Dialogue:\n{review['dialogue']}")
                
                moderator_prompt = f"""As a scientific moderator, analyze the complete review discussion and debate:

{' '.join(all_dialogue)}

Please provide:
1. Evolution of Discussion and Debate Points
2. Review Quality Assessment
3. Areas of Consensus and Disagreement
4. Key Points Synthesis
5. Final Recommendation with Score Assessment"""

                moderator_response = moderator_agent.invoke([HumanMessage(content=moderator_prompt)])
                st.markdown(moderator_response.content)
            except Exception as e:
                st.error(f"Error in moderator analysis: {str(e)}")
        
        # Iteration summaries
        st.subheader("Review History")
        for i, iteration in enumerate(all_reviews, 1):
            with st.expander(f"Iteration {i}", expanded=False):
                for review in iteration:
                    if review["success"]:
                        st.markdown(f"**Review by {review['expertise']['name']}**")
                        st.markdown(review['review'])
                        if review.get('dialogue'):
                            st.markdown("**Dialogue Response:**")
                            st.markdown(review['dialogue'])

        # Export options
        pdf_bytes = generate_pdf_summary(all_reviews, scores)
        st.download_button(
            label="Download Complete Review Summary (PDF)",
            data=pdf_bytes,
            file_name="review_summary.pdf",
            mime="application/pdf",
        )

    status_text.empty()
    progress_bar.empty()
    return {"reviews": review_results, "success": True}
def get_expert_dialogue_prompt(reviews: List[Dict], expertise: str, rating_scale: str) -> str:
    return f"""As {expertise}, analyze and respond to the other reviews:

Previous reviews:
{' '.join([r['review'] for r in reviews if r['success'] and r['expertise']['name'] != expertise])}

Please:
1. Address other reviewers' critiques
2. Defend or revise your assessments
3. Highlight areas of agreement/disagreement
4. Provide evidence for your positions
5. Update your score using the {rating_scale} scale if needed"""

def get_iteration_prompt(expertise: str, iteration: int, previous_reviews: List[Dict], topic: str, rating_scale: str) -> str:
    prompt = f"""As an expert in {expertise}, this is iteration {iteration} of the review.

Previous discussion history:
"""
    for rev in previous_reviews:
        if rev.get("dialogue"):
            prompt += f"\n{rev['expertise']['name']} Review:\n{rev['review']}\n"
            prompt += f"Dialogue:\n{rev['dialogue']}\n"
        else:
            prompt += f"\n{rev['expertise']['name']} Review:\n{rev['review']}\n"

    prompt += f"""
Please provide your {'updated' if iteration > 1 else 'initial'} review considering the above discussion:
1. Overview and Analysis
2. Response to Previous Reviews
3. Methodology Assessment
4. Key Points of Agreement/Disagreement
5. Recommendations
6. Score using {rating_scale}"""

    return prompt

def generate_pdf_summary(all_reviews: List[List[Dict]], scores: List[float]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Add custom style for dialogue
    dialogue_style = ParagraphStyle(
        name='Dialogue',
        parent=styles['Normal'],
        leftIndent=20,
        textColor=colors.blue,
        spaceAfter=12
    )
    styles.add(dialogue_style)
    
    story = []
    story.append(Paragraph("Scientific Review Summary", styles['Title']))
    story.append(Spacer(1, 12))
    
    if scores:
        avg_score = sum(scores) / len(scores)
        story.append(Paragraph(f"Overall Score: {avg_score:.2f}", styles['Heading1']))
        description = get_score_description(st.session_state.get('rating_scale', 'Paper Score (-2 to 2)'), avg_score)
        story.append(Paragraph(f"Assessment: {description}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    for iteration_idx, iteration in enumerate(all_reviews, 1):
        story.append(Paragraph(f"Iteration {iteration_idx}", styles['Heading1']))
        for review in iteration:
            if review["success"]:
                story.append(Paragraph(f"Review by {review['expertise']['name']}", styles['Heading2']))
                story.append(Paragraph(f"Model: {review['expertise']['model']}", styles['Normal']))
                story.append(Paragraph(f"Style: {review['expertise']['style']}", styles['Normal']))
                story.append(Paragraph(review['review'], styles['Normal']))
                if review.get('dialogue'):
                    story.append(Paragraph("Expert Dialogue Response:", styles['Heading3']))
                    story.append(Paragraph(review['dialogue'], dialogue_style))
                story.append(Spacer(1, 12))
    
    # Generate and include moderator analysis in PDF
    moderator_analysis = ""
    if len(agents) > len(expertises):
        try:
            moderator_agent = agents[-1]
            all_dialogue = []
            for iteration in all_reviews:
                for review in iteration:
                    if review["success"]:
                        all_dialogue.append(f"{review['expertise']['name']} Review:\n{review['review']}")
                        if review.get('dialogue'):
                            all_dialogue.append(f"Dialogue:\n{review['dialogue']}")
            
            moderator_prompt = f"""As a scientific moderator, analyze the complete review discussion:

{' '.join(all_dialogue)}

Please provide:
1. Evolution of Discussion
2. Review Quality Assessment
3. Key Points Synthesis
4. Overall Recommendation
5. Final Score Assessment"""

            moderator_response = moderator_agent.invoke([HumanMessage(content=moderator_prompt)])
            moderator_analysis = moderator_response.content
            
            # Add moderator analysis to PDF
            story.append(Paragraph("Moderator Analysis", styles['Heading1']))
            story.append(Paragraph(moderator_analysis, styles['Normal']))
            story.append(Spacer(1, 12))
            
        except Exception as e:
            story.append(Paragraph(f"Error in moderator analysis: {str(e)}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

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
        process_review_memoryless(
            content=content,
            agents=agents,
            expertises=expertises,
            custom_prompts=custom_prompts,
            num_iterations=num_iterations
        )

if __name__ == "__main__":
    scientific_review_page()
