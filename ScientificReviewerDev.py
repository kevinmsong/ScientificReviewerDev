import streamlit as st
import logging
from openai import OpenAI
# Updated import for ChatOpenAI:
from langchain.chat_models import ChatOpenAI  
# Updated import for HumanMessage:
from langchain.schema import HumanMessage  
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
    
    if rating_scale == "Paper Score (-2 to 2)":
        score_range = (-2, 2)
        step = 1
    elif rating_scale == "Star Rating (1-5)":
        score_range = (1, 5)
        step = 1
    else:  # NIH Scale
        score_range = (1, 9)
        step = 2
    
    min_score, max_score = score_range
    normalized_score = min(max(score, min_score), max_score)
    rounded_score = round(normalized_score / step) * step
    
    scale_mapping = descriptions.get(rating_scale, {})
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
    
    if include_moderator:
        moderator_agent = ChatOpenAI(temperature=0.1, openai_api_key=st.secrets["openai_api_key"], model="gpt-4o")
        agents.append(moderator_agent)
    
    return agents

def get_dialogue_prompt(current_expert: str, all_reviews: List[Dict], rating_scale: str, content: str) -> str:
    other_reviews = [r for r in all_reviews if r['expertise']['name'] != current_expert]
    review_history = "\n".join([
        f"Review by {r['expertise']['name']}:\n{r['review']}"
        for r in other_reviews
    ])
    
    scale_info = {
        "Paper Score (-2 to 2)": "using -2 to 2 scale",
        "Star Rating (1-5)": "using 1-5 stars",
        "NIH Scale (1-9)": "using 1 (exceptional) to 9 (poor)"
    }
    
    return f"""Document Content:
{content}

Review History:
{review_history}

As {current_expert}, provide:
1. Specific responses to points in other reviews
2. Areas of agreement/disagreement with assessments
3. Defense or revision of your positions 
4. Updated scores {scale_info[rating_scale]}
5. Specific questions for other reviewers"""



def join_reviews(reviews: List[Dict]) -> str:
    return "\n".join([
        f"{r['expertise']['name']}: {r['review']}\n" +
        (f"Dialogue: {r.get('dialogue', '')}\n" if r.get('dialogue') else "")
        for r in reviews
    ])

def get_moderator_summary(reviews: List[Dict], final_iteration: bool = False) -> str:
    review_history = join_reviews(reviews)
    
    if final_iteration:
        sections = [
            "Final consensus and key findings",
            "Overall recommendation and score synthesis",
            "Quality and completeness of peer review process",
            "Impact and significance assessment"
        ]
    else:
        sections = [
            "Current status of discussion",
            "Emerging patterns and disagreements",
            "Suggestions for next iteration",
            "Areas needing more discussion"
        ]
    
    summary_type = "final" if final_iteration else "interim"
    prompt = f"""As a scientific moderator, provide a {summary_type} analysis of the review discussion:

Review History:
{review_history}

Please provide:
1. {sections[0]}
2. {sections[1]} 
3. {sections[2]}
4. {sections[3]}"""

    return prompt

def get_default_prompt(review_type: str, expertise: str) -> str:
    prompts = {
        "Paper": f"""As {expertise}, review this paper considering:
1. Scientific Merit & Innovation
2. Methodology & Technical Rigor
3. Data Analysis & Results
4. Clarity & Organization
5. Impact & Significance""",
        "Grant": f"""As {expertise}, evaluate this grant proposal considering:
1. Innovation & Merit
2. Methodology & Feasibility
3. Resources & Budget
4. Timeline & Deliverables
5. Impact & Significance""",
        "Poster": f"""As {expertise}, review this poster considering:
1. Visual Design & Organization
2. Content & Clarity
3. Technical Merit
4. Results & Discussion
5. Overall Impact"""
    }
    return prompts.get(review_type, f"Review this {review_type.lower()}")

def adjust_prompt_style(prompt: str, style: int, rating_scale: str) -> str:
    styles = {
        -2: "Be extremely thorough and critical in your assessment.",
        -1: "Maintain high academic standards with careful attention to methodology.",
        0: "Provide balanced evaluation of strengths and weaknesses.",
        1: "Focus on strengths while noting necessary improvements.",
        2: "Take an encouraging approach while maintaining scientific rigor."
    }
    
    scale_info = {
        "Paper Score (-2 to 2)": "Score each section and overall paper from -2 (worst) to 2 (best)",
        "Star Rating (1-5)": "Rate each section and overall paper from 1-5 stars",
        "NIH Scale (1-9)": "Score each section and overall paper from 1 (exceptional) to 9 (poor)"
    }
    
    return f"{prompt}\n\nApproach: {styles[style]}\nScoring: {scale_info[rating_scale]}"

def process_chunk_memoryless(chunk: str, agent: Union[ChatOpenAI, Any], expertise: str, prompt: str, model_type: str) -> str:
    logging.info(f"Processing chunk for {expertise}")
    logging.info(f"Chunk preview: {chunk[:200]}...")
    
    try:
        if model_type == "GPT-4o":
            response = agent.invoke([HumanMessage(content=prompt + "\n\nDocument Content:\n" + chunk)])
            return response.content
        else:
            response = agent.generate_content(prompt + "\n\nDocument Content:\n" + chunk)
            return response.text
    except Exception as e:
        logging.error(f"Error processing chunk for {expertise}: {str(e)}")
        return f"[Error: {str(e)}]"

def extract_pdf_content(pdf_file) -> Tuple[str, List[Image.Image]]:
    logging.info("Starting PDF extraction")
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text_content = ""
    images = []
    
    for page_num, page in enumerate(pdf_document):
        page_text = page.get_text()
        text_content += page_text
        logging.info(f"Extracted page {page_num + 1}, length: {len(page_text)} chars")
        
        for img in page.get_images():
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
            logging.info(f"Extracted image from page {page_num + 1}")
    
    logging.info(f"Total extracted text length: {len(text_content)} chars")
    return text_content, images

def process_review_memoryless(content: str,
                              agents: List[Union[ChatOpenAI, Any]],
                              expertises: List[Dict],
                              custom_prompts: List[str],
                              *,
                              num_iterations: int = 1) -> Dict[str, Any]:
    all_reviews = []
    scores = []
    previous_iteration = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    tabs = st.tabs([f"Iteration {i+1}" for i in range(num_iterations)] + ["Final Analysis"])
    
    total_steps = num_iterations * (len(agents) * 2 + 1)
    current_step = 0
    
    # If a moderator is included, it is added as the last agent.
    moderator_agent = agents[-1] if len(agents) > len(expertises) else None
    review_agents = agents[:-1] if moderator_agent else agents
    
    for iteration in range(num_iterations):
        iteration_reviews = []
        with tabs[iteration]:
            st.subheader(f"Iteration {iteration + 1}")
            
            # Process individual reviews
            for i, (agent, expertise, base_prompt) in enumerate(zip(review_agents, expertises, custom_prompts)):
                status_text.text(f"Processing review from {expertise['name']} (Iteration {iteration + 1})")
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                try:
                    review_prompt = get_iteration_prompt(expertise['name'], iteration + 1, previous_iteration, base_prompt)
                    review_text = process_chunk_memoryless(content, agent, expertise['name'], review_prompt, expertise['model'])
                    
                    review_data = {
                        "expertise": expertise,
                        "review": review_text,
                        "success": True,
                        "iteration": iteration + 1
                    }
                    iteration_reviews.append(review_data)
                    
                    with st.expander(f"Review by {expertise['name']}", expanded=True):
                        st.markdown(review_text)
                
                except Exception as e:
                    st.error(f"Error in review: {str(e)}")
                    iteration_reviews.append({
                        "expertise": expertise,
                        "review": f"Error: {str(e)}",
                        "success": False,
                        "iteration": iteration + 1
                    })
            
            # Process expert dialogue if more than one reviewer is present
            if len(expertises) > 1:
                st.subheader("Expert Dialogue")
                dialogue_container = st.container()
                
                for i, review in enumerate(iteration_reviews):
                    if review["success"]:
                        status_text.text(f"Processing dialogue for {review['expertise']['name']}")
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                        
                        dialogue_prompt = get_dialogue_prompt(
                            review['expertise']['name'], 
                            iteration_reviews, 
                            st.session_state['rating_scale'],
                            content
                        )
                        # Compute "other reviews" for context by excluding the current review
                        other_reviews = [r for j, r in enumerate(iteration_reviews) if j != i]
                        
                        dialogue = process_expert_dialogue(
                            agent=review_agents[i],
                            dialogue_prompt=dialogue_prompt,
                            model_type=review['expertise']['model'],
                            current_review=review,
                            other_reviews=other_reviews
                        )
                        
                        review['dialogue'] = dialogue
                        with dialogue_container:
                            with st.expander(f"Response from {review['expertise']['name']}", expanded=True):
                                st.markdown(dialogue)
            
            # Generate a moderation summary if a moderator agent is included
            if moderator_agent:
                status_text.text("Generating moderation summary...")
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                moderation = run_moderation(moderator_agent, iteration_reviews, final_iteration=(iteration == num_iterations - 1))
                st.subheader("Moderation Summary")
                st.markdown(moderation)
            
            all_reviews.append(iteration_reviews)
            previous_iteration = iteration_reviews.copy()
            
            # Extract scores from reviews and dialogue if available
            for review in iteration_reviews:
                if review["success"]:
                    score_matches = re.findall(r'score[:\s]*(-?\d+\.?\d*)', review['review'].lower())
                    score_matches.extend(re.findall(r'score[:\s]*(-?\d+\.?\d*)', review.get('dialogue', '').lower()))
                    if score_matches:
                        try:
                            scores.append(float(score_matches[-1]))
                        except ValueError:
                            pass

    with tabs[-1]:
        create_final_analysis(all_reviews, scores, moderator_agent)
    
    status_text.empty()
    progress_bar.empty()
    return {"reviews": all_reviews, "success": True}


def create_final_analysis(all_reviews: List[List[Dict]], scores: List[float], moderator_agent: Union[ChatOpenAI, Any] = None):
    st.subheader("Review Summary")
    
    if scores:
        avg_score = sum(scores) / len(scores)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Score", f"{avg_score:.2f}")
        with col2:
            st.write("Assessment:", get_score_description(st.session_state['rating_scale'], avg_score))
    
    if moderator_agent:
        st.subheader("Final Moderator Analysis")
        final_reviews = [review for iteration in all_reviews for review in iteration]
        final_analysis = run_moderation(moderator_agent, final_reviews, final_iteration=True)
        st.markdown(final_analysis)
    
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
    
    pdf_bytes = generate_pdf_summary(all_reviews, scores, final_analysis if moderator_agent else None)
    st.download_button(
        label="Download Complete Review Summary (PDF)",
        data=pdf_bytes,
        file_name="review_summary.pdf",
        mime="application/pdf"
    )

def generate_pdf_summary(all_reviews: List[List[Dict]], scores: List[float], final_analysis: str = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(
        name='ReviewHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12
    ))
    
    styles.add(ParagraphStyle(
        name='ReviewBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    ))
    
    styles.add(ParagraphStyle(
        name='DialogueText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        textColor=colors.HexColor('#2020CC'),
        spaceAfter=12
    ))
    
    story = []
    story.append(Paragraph("Scientific Review Summary", styles['Title']))
    story.append(Spacer(1, 24))
    
    if scores:
        story.append(Paragraph("Overall Assessment", styles['Heading1']))
        avg_score = sum(scores) / len(scores)
        story.append(Paragraph(f"Score: {avg_score:.2f}", styles['ReviewBody']))
        description = get_score_description(st.session_state['rating_scale'], avg_score)
        story.append(Paragraph(f"Assessment: {description}", styles['ReviewBody']))
        story.append(Spacer(1, 24))
    
    for iteration_idx, iteration in enumerate(all_reviews, 1):
        story.append(Paragraph(f"Iteration {iteration_idx}", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        for review in iteration:
            if review["success"]:
                story.append(Paragraph(f"Review by {review['expertise']['name']}", styles['ReviewHeader']))
                story.extend(convert_markdown_to_reportlab(review['review'], styles))
                
                if review.get('dialogue'):
                    story.append(Spacer(1, 12))
                    story.append(Paragraph("Expert Dialogue:", styles['Heading3']))
                    story.extend(convert_markdown_to_reportlab(review['dialogue'], styles))
                story.append(Spacer(1, 24))
    
    if final_analysis:
        story.append(Paragraph("Final Analysis", styles['Heading1']))
        story.extend(convert_markdown_to_reportlab(final_analysis, styles))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def scientific_review_page():
    st.set_page_config(page_title="Scientific Reviewer", layout="wide")
    st.header("Scientific Review System")
    st.caption("v2.3.0 - Enhanced Dialogue Mode")
    
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
        num_iterations = st.number_input("Number of Discussion Iterations", 1, 5, 1)
    
    use_moderator = st.checkbox("Include Moderator", value=True) if num_reviewers > 1 else False
    
    expertises = []
    custom_prompts = []
    
    with st.expander("Configure Reviewers", expanded=True):
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

def process_list(items: List[str], styles: Dict) -> List[Paragraph]:
    if not items:
        return []
    bullet_style = ParagraphStyle(
        'BulletPoint',
        parent=styles['ReviewBody'],
        leftIndent=20,
        firstLineIndent=0,
        bulletIndent=10,
        spaceBefore=3,
        spaceAfter=3
    )
    return [Paragraph(f"• {item}", bullet_style) for item in items]

def convert_markdown_to_reportlab(text: str, styles: Dict) -> List[Paragraph]:
    paragraphs = []
    current_list = []
    in_list = False
    
    for line in text.split('\n'):
        # Headers
        if line.startswith('# '):
            if current_list:
                paragraphs.extend(process_list(current_list, styles))
                current_list = []
            in_list = False
            paragraphs.append(Paragraph(line[2:], styles['Heading1']))
        elif line.startswith('## '):
            if current_list:
                paragraphs.extend(process_list(current_list, styles))
                current_list = []
            in_list = False
            paragraphs.append(Paragraph(line[3:], styles['Heading2']))
        elif line.startswith('### '):
            if current_list:
                paragraphs.extend(process_list(current_list, styles))
                current_list = []
            in_list = False
            paragraphs.append(Paragraph(line[4:], styles['Heading3']))
        # Lists
        elif line.strip().startswith(('* ', '- ', '+')):
            in_list = True
            current_list.append(line.strip()[2:])
        elif re.match(r'\d+\.', line.strip()):
            in_list = True
            current_list.append(line.strip().split('.', 1)[1].strip())
        # Regular text
        elif line.strip():
            if in_list:
                paragraphs.extend(process_list(current_list, styles))
                current_list = []
                in_list = False
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            line = re.sub(r'_(.*?)_', r'<i>\1</i>', line)
            paragraphs.append(Paragraph(line, styles['ReviewBody']))
        else:
            if current_list:
                paragraphs.extend(process_list(current_list, styles))
                current_list = []
                in_list = False
            paragraphs.append(Spacer(1, 6))
    
    if current_list:
        paragraphs.extend(process_list(current_list, styles))
    
    return paragraphs

def run_moderation(agent: Union[ChatOpenAI, Any], reviews: List[Dict], final_iteration: bool = False) -> str:
    try:
        prompt = get_moderator_summary(reviews, final_iteration)
        response = agent.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logging.error(f"Error in moderation: {str(e)}")
        return f"[Error in moderation: {str(e)}]"

def get_iteration_prompt(expertise: str, iteration: int, previous_reviews: List[Dict], base_prompt: str) -> str:
    if iteration == 1 or not previous_reviews:
        return base_prompt
    
    history = "\n\n".join([
        f"{rev['expertise']['name']} Review:\n{rev['review']}\n" +
        (f"Dialogue:\n{rev['dialogue']}\n" if rev.get('dialogue') else "")
        for rev in previous_reviews
    ])
    
    return f"""{base_prompt}

Previous Discussion History:
{history}

Please provide an updated review considering the above discussion."""

def process_expert_dialogue(agent: Union[ChatOpenAI, Any], dialogue_prompt: str, model_type: str, current_review: Dict, other_reviews: List[Dict]) -> str:
    review_quotes = "\n\n".join([
        f"Review by {r['expertise']['name']}:\n" + 
        "\n".join(f"> {line}" for line in r['review'].split('\n') if line.strip())
        for r in other_reviews
    ])
    
    if model_type == "GPT-4o":
        final_prompt = dialogue_prompt
    else:
        final_prompt = f"""Response to Other Reviews of Paper

These are the actual reviews to address:
{review_quotes}

Your previous review was:
{current_review['review']}

REQUIREMENTS:
- Respond directly to points in the reviews quoted above
- Quote specific text when addressing reviewer points
- Stay factual and grounded in the actual reviews
- Use the exact scoring scale specified
- No hypothetical or imagined reviewer comments

{dialogue_prompt}"""
    
    try:
        if model_type == "GPT-4o":
            response = agent.invoke([HumanMessage(content=final_prompt)])
            return response.content
        else:
            response = agent.generate_content(final_prompt)
            return response.text
    except Exception as e:
        logging.error(f"Error in dialogue: {str(e)}")
        return f"[Error in dialogue: {str(e)}]"
    
if __name__ == "__main__":
    scientific_review_page()
