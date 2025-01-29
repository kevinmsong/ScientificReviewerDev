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
import time
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)

# Initialize API clients
openai_api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=openai_api_key)

def create_review_agents(expertises: List[Dict], review_type: str = "paper", include_moderator: bool = False) -> List[Union[ChatOpenAI, Any]]:
    agents = []
    
    for expertise in expertises:
        if expertise["model"] == "GPT-4o":
            agent = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key, model="gpt-4o")
        else:
            genai.configure(api_key=st.secrets["gemini_api_key"])
            agent = genai.GenerativeModel("gemini-2.0-flash-exp")
        agents.append(agent)
    
    if include_moderator and len(expertises) > 1:
        moderator_agent = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key, model="gpt-4o")
        agents.append(moderator_agent)
    
    return agents

def chunk_content(text: str, max_tokens: int = 100000) -> List[str]:
    """Split content into chunks that fit within token limits."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph_tokens = encoding.encode(paragraph)
        if len(paragraph_tokens) > max_tokens:
            sentences = paragraph.split('. ')
            for sentence in sentences:
                sentence_tokens = encoding.encode(sentence)
                if current_length + len(sentence_tokens) > max_tokens:
                    if current_chunk:
                        chunks.append(encoding.decode(current_chunk))
                        current_chunk = []
                        current_length = 0
                current_chunk.extend(sentence_tokens)
                current_length += len(sentence_tokens)
        else:
            if current_length + len(paragraph_tokens) > max_tokens:
                chunks.append(encoding.decode(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.extend(paragraph_tokens)
            current_length += len(paragraph_tokens)
    
    if current_chunk:
        chunks.append(encoding.decode(current_chunk))
    
    return chunks

def extract_content(response: Union[str, Any], default_value: str) -> str:
    """Extract content from various response types."""
    if isinstance(response, str):
        return response
    elif hasattr(response, 'content'):
        return response.content
    elif isinstance(response, list) and len(response) > 0:
        return response[0].content
    else:
        logging.warning(f"Unexpected response type: {type(response)}")
        return default_value

def extract_pdf_content(pdf_file) -> Tuple[str, List[Image.Image]]:
    """Extract text and images from a PDF file."""
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

def get_debate_prompt(expertise: str, iteration: int, previous_reviews: List[Dict[str, str]], topic: str) -> str:
    """Generate a debate-style prompt for reviewers."""
    prompt = f"""As an expert in {expertise}, you are participating in iteration {iteration} of a scientific review discussion.

Previous reviews and comments to consider:

"""
    for prev_review in previous_reviews:
        prompt += f"\nReview by {prev_review['expertise']}:\n{prev_review['review']}\n"
        
    if iteration == 1:
        prompt += f"""
Please provide your initial review of this {topic} with:
1. Overview and Summary
2. Technical Analysis
3. Methodology Assessment
4. Strengths
5. Weaknesses
6. Suggestions for Improvement
7. Scores (1-9)
"""
    else:
        prompt += f"""
Based on the previous reviews, please:
1. Address points raised by other reviewers
2. Defend or revise your previous assessments
3. Identify areas of agreement and disagreement
4. Provide additional insights or counterpoints
5. Update your scores if necessary
"""
    return prompt

def process_chunks_with_debate(chunks: List[str], agent: Union[ChatOpenAI, Any], expertise: str, 
                             prompt: str, iteration: int, model_type: str = "GPT-4o") -> str:
    chunk_reviews = []
    
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"""Reviewing part {i+1} of {len(chunks)}:
{prompt}
Content part {i+1}/{len(chunks)}:
{chunk}"""

        try:
            if model_type == "GPT-4o":
                response = agent.invoke([HumanMessage(content=chunk_prompt)])
                chunk_review = extract_content(response, f"[Error processing chunk {i+1}]")
            else:
                response = agent.generate_content(chunk_prompt)
                chunk_review = response.text
            chunk_reviews.append(chunk_review)
        except Exception as e:
            logging.error(f"Error processing chunk {i+1} for {expertise}: {str(e)}")
            chunk_reviews.append(f"[Error in chunk {i+1}]")
    
    if len(chunks) > 1:
        compilation_prompt = f"""Compile your reviews of all {len(chunks)} parts:
{''.join(chunk_reviews)}"""

        try:
            if model_type == "GPT-4o":
                compilation_response = agent.invoke([HumanMessage(content=compilation_prompt)])
                return extract_content(compilation_response, "[Error compiling final review]")
            else:
                compilation_response = agent.generate_content(compilation_prompt)
                return compilation_response.text
        except Exception as e:
            logging.error(f"Error compiling review for {expertise}: {str(e)}")
            return "\n\n".join(chunk_reviews)
    
    return chunk_reviews[0]

def generate_debate_summary(reviews: List[Dict], expertise: str, rating_scale: str) -> str:
    """Generate a summary prompt for expert dialogue."""
    summary = "Previous reviews for discussion:\n\n"
    for review in reviews:
        if review["success"]:
            summary += f"Review by {review['expertise']['name']}:\n"
            summary += f"{review['review']}\n\n"
    
    scale_info = {
        "Paper Score (-2 to 2)": "(-2: worst, 2: best)",
        "Star Rating (1-5)": "(1-5 stars)",
        "NIH Scale (1-9)": "(1: exceptional, 9: poor)"
    }
    
    prompt = f"""As {expertise}, analyze the reviews and provide:

1. Response to Reviews
- Address key points and critiques
- Discuss methodology assessments
- Evaluate conclusions

2. Comparative Analysis
- Areas of agreement/disagreement
- Evidence assessment
- Methodology considerations

3. Final Assessment
- Updated evaluation using {rating_scale} {scale_info[rating_scale]}
- Recommendations
- Critical considerations

Base all responses on evidence and specific points from the reviews."""
    
    return summary + prompt

def process_reviews_with_debate(content: str, agents: List[Union[ChatOpenAI, Any]], expertises: List[Dict], 
                              custom_prompts: List[str], review_type: str, num_iterations: int, 
                              progress_callback=None) -> Dict[str, Any]:
    all_iterations = []
    latest_reviews = []
    tabs = st.tabs([f"Iteration {i+1}" for i in range(num_iterations)] + ["Moderator Analysis"])
    
    for iteration in range(num_iterations):
        with tabs[iteration]:
            st.write(f"Starting iteration {iteration + 1}")
            review_results = []
            
            # Initial reviews
            for i, (agent, expertise, base_prompt) in enumerate(zip(agents[:-1] if len(agents) > len(expertises) else agents, expertises, custom_prompts)):
                review_container = st.container()
                with review_container:
                    processing_msg = st.empty()
                    processing_msg.info(f"Processing review from {expertise['name']}...")
                    try:
                        debate_prompt = get_debate_prompt(expertise['name'], iteration + 1, latest_reviews, review_type)
                        full_prompt = f"{base_prompt}\n\n{debate_prompt}"
                        
                        review_text = process_chunks_with_debate(
                            chunks=chunk_content(content),
                            agent=agent,
                            expertise=expertise,
                            prompt=full_prompt,
                            iteration=iteration + 1,
                            model_type=expertise['model']
                        )
                        
                        review_results.append({
                            "expertise": expertise,
                            "review": review_text,
                            "iteration": iteration + 1,
                            "success": True
                        })
                        
                        processing_msg.empty()
                        with st.expander(f"Review by {expertise['name']} ({expertise['model']})", expanded=True):
                            st.markdown(review_text)
                            st.caption(f"Critique Style: {expertise['style']}")
                            
                    except Exception as e:
                        logging.error(f"Error processing agent {expertise}: {str(e)}")
                        review_results.append({
                            "expertise": expertise,
                            "review": f"Error: {str(e)}",
                            "iteration": iteration + 1,
                            "success": False
                        })
                        processing_msg.error(f"Error processing review from {expertise['name']}")
            
            # Expert dialogue
            st.subheader("Expert Dialogue")
            for expertise, agent in zip(expertises, agents[:-1] if len(agents) > len(expertises) else agents):
                try:
                    dialogue_prompt = generate_debate_summary(review_results, expertise['name'])
                    if expertise['model'] == "GPT-4o":
                        response = agent.invoke([HumanMessage(content=dialogue_prompt)])
                        dialogue = extract_content(response, "[Error in dialogue]")
                    else:
                        response = agent.generate_content(dialogue_prompt)
                        dialogue = response.text
                        
                    with st.expander(f"Response from {expertise['name']}", expanded=True):
                        st.markdown(dialogue)
                        
                    review_results[next(i for i, r in enumerate(review_results) if r["expertise"]["name"] == expertise["name"])]["dialogue"] = dialogue
                        
                except Exception as e:
                    st.error(f"Error in dialogue for {expertise['name']}: {str(e)}")
            
            all_iterations.append(review_results)
            latest_reviews = review_results
            st.success(f"Completed iteration {iteration + 1}")

    # Moderator analysis
    if len(agents) > len(expertises):
        with tabs[-1]:
            st.subheader("Moderator Analysis")
            moderator_agent = agents[-1]
            
            try:
                analysis = generate_moderator_analysis(all_iterations)
                if isinstance(moderator_agent, ChatOpenAI):
                    response = moderator_agent.invoke([HumanMessage(content=analysis)])
                    moderation = extract_content(response, "[Error in moderation]")
                else:
                    response = moderator_agent.generate_content(analysis)
                    moderation = response.text
                
                st.markdown(moderation)
                
            except Exception as e:
                st.error(f"Error in moderation: {str(e)}")

    return {
        "all_iterations": all_iterations,
        "success": True
    }

def generate_moderator_analysis(all_iterations: List[List[Dict]]) -> str:
    summary = "Complete review discussion for analysis:\n\n"
    
    for iteration_idx, iteration_reviews in enumerate(all_iterations, 1):
        summary += f"\nIteration {iteration_idx}:\n"
        for review in iteration_reviews:
            if review.get("success", False):
                summary += f"\nReview by {review['expertise']['name']}:\n{review['review']}\n"
                if "dialogue" in review:
                    summary += f"\nDialogue contribution:\n{review['dialogue']}\n"
    
    prompt = """As a scientific moderator, provide a comprehensive analysis:

1. Evolution of Discussion
- How perspectives evolved across iterations
- Key points of agreement/disagreement
- Quality of scientific discourse

2. Review Quality Assessment
- Rigor of arguments
- Evidence quality
- Constructiveness of dialogue

3. Moderator Synthesis
- Critical consensus points
- Unresolved debates
- Priority recommendations

4. Recommendation
- Overall assessment
- Decision recommendation
- Key action items"""
    
    return summary + "\n\n" + prompt

def adjust_prompt_style(prompt: str, style: int) -> str:
    """Adjust the prompt based on critique style."""
    style_adjustments = {
        -2: """Please be extremely thorough and critical in your review. Focus heavily on identifying weaknesses, methodological flaws, and areas needing significant improvement. While acknowledging strengths, emphasize rigorous critique.""",
        -1: """Maintain high standards in your review. Carefully identify both strengths and weaknesses, with particular attention to areas needing improvement.""",
        0: """Provide a balanced review that equally considers both strengths and weaknesses. Aim for constructive feedback that acknowledges both positive aspects and areas for improvement.""",
        1: """While identifying areas for improvement, emphasize the positive aspects and potential of the work. Focus on constructive suggestions rather than criticism.""",
        2: """Take an encouraging and supportive approach in your review. While noting any critical issues, focus primarily on strengths and positive aspects of the work."""
    }
    
    return f"{prompt}\n\nReview Style: {style_adjustments[style]}"

def generate_moderator_prompt(all_iterations: List[List[Dict[str, str]]]) -> str:
    """Generate the prompt for the moderator's analysis."""
    prompt = """As a senior scientific moderator, analyze the complete review discussion:

"""
    for iteration_idx, iteration_reviews in enumerate(all_iterations, 1):
        prompt += f"\nIteration {iteration_idx}:\n"
        for review in iteration_reviews:
            if review.get("success", False):
                prompt += f"\nReview by {review['expertise']}:\n{review['review']}\n"
    
    prompt += """
Please provide:
1. Discussion Evolution
2. Review Analysis
3. Key Points Synthesis
4. Moderator Assessment including scores and recommendation
"""
    return prompt

def get_default_prompt(review_type: str, expertise: str) -> str:
    """Get default prompt based on review type."""
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

def scientific_review_page():
    try:
        st.set_page_config(page_title="Scientific Reviewer", layout="wide")
        st.header("Scientific Review System")
        st.caption("v2.1.0")
        
        col1, col2 = st.columns([2,1])
        with col1:
            rating_scale = st.radio(
                "Rating Scale",
                ["Paper Score (-2 to 2)", "Star Rating (1-5)", "NIH Scale (1-9)"],
                help="Paper: -2 (worst) to 2 (best)\nStar: 1-5 stars\nNIH: 1 (best) to 9 (worst)"
            )
        
        def get_score_description(scale: str, score: Union[int, float]) -> str:
            descriptions = {
                "Paper Score (-2 to 2)": {
                    -2: "Major flaws, rejection recommended",
                    -1: "Significant revisions needed",
                    0: "Moderate revisions needed",
                    1: "Minor revisions needed",
                    2: "Accept as is"
                },
                "Star Rating (1-5)": {
                    1: "Poor",
                    2: "Fair",
                    3: "Good",
                    4: "Very Good",
                    5: "Excellent"
                },
                "NIH Scale (1-9)": {
                    1: "Exceptional",
                    2: "Outstanding",
                    3: "Excellent",
                    4: "Very Good",
                    5: "Good",
                    6: "Satisfactory",
                    7: "Fair",
                    8: "Marginal",
                    9: "Poor"
                }
            }
            return descriptions[scale].get(score, "")
        
        def adjust_prompt_style(prompt: str, style: int, rating_scale: str) -> str:
            """Adjust prompt based on critique style and rating scale."""
            style_adjustments = {
                -2: "Be extremely thorough and critical. Focus on weaknesses and flaws.",
                -1: "Maintain high standards. Carefully identify both strengths and weaknesses.",
                0: "Provide balanced review of strengths and weaknesses.",
                1: "Emphasize positive aspects while noting necessary improvements.",
                2: "Take an encouraging approach while noting critical issues."
            }
            
            scale_instructions = {
                "Paper Score (-2 to 2)": "Score from -2 (worst) to 2 (best)",
                "Star Rating (1-5)": "Rate from 1 to 5 stars",
                "NIH Scale (1-9)": "Score from 1 (exceptional) to 9 (poor)"
            }
            
            return f"{prompt}\n\nReview Style: {style_adjustments[style]}\n\nRating: {scale_instructions[rating_scale]}"

if __name__ == "__main__":
    try:
        scientific_review_page()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logging.exception("Error in main:")
