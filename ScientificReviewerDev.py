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

def create_review_agents(num_agents: int, review_type: str = "paper", include_moderator: bool = False, model_type: str = "GPT-4o") -> List[Union[ChatOpenAI, Any]]:
    agents = []
    
    if model_type == "GPT-4o":
        model = "gpt-4o"
        for _ in range(num_agents):
            agents.append(ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key, model=model))
    else:
        genai.configure(api_key=st.secrets["gemini_api_key"])
        model = genai.GenerativeModel("gemini-2.0-flash")
        for _ in range(num_agents):
            agents.append(model)
    
    if include_moderator and num_agents > 1:
        if model_type == "GPT-4o":
            moderator_agent = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key, model="gpt-4o")
        else:
            moderator_agent = model
        agents.append(moderator_agent)
    
    return agents

def chunk_content(text: str, max_tokens: int = 6000) -> List[str]:
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

def process_reviews_with_debate(content: str, agents: List[Union[ChatOpenAI, Any]], expertises: List[str], 
                              custom_prompts: List[str], review_type: str, num_iterations: int, 
                              model_type: str = "GPT-4o", progress_callback=None) -> Dict[str, Any]:
    logging.info(f"Starting process_reviews_with_debate")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Number of agents: {len(agents)}")
    
    all_iterations = []
    latest_reviews = []
    
    for iteration in range(num_iterations):
        review_results = []
        st.write(f"Starting iteration {iteration + 1}")
        
        for i, (agent, expertise, base_prompt) in enumerate(zip(agents[:-1] if len(agents) > len(expertises) else agents, expertises, custom_prompts)):
            processing_msg = st.empty()
            processing_msg.info(f"Processing review from {expertise}...")
            try:
                debate_prompt = get_debate_prompt(expertise, iteration + 1, latest_reviews, review_type)
                full_prompt = f"{base_prompt}\n\n{debate_prompt}"
                
                review_text = process_chunks_with_debate(
                    chunks=chunk_content(content),
                    agent=agent,
                    expertise=expertise,
                    prompt=full_prompt,
                    iteration=iteration + 1,
                    model_type=model_type
                )
                
                review_results.append({
                    "expertise": expertise,
                    "review": review_text,
                    "iteration": iteration + 1,
                    "success": True
                })
                processing_msg.success(f"Completed review from {expertise}")
                
            except Exception as e:
                logging.error(f"Error processing agent {expertise}: {str(e)}")
                review_results.append({
                    "expertise": expertise,
                    "review": f"Error: {str(e)}",
                    "iteration": iteration + 1,
                    "success": False
                })
                processing_msg.error(f"Error processing review from {expertise}")
        
        all_iterations.append(review_results)
        latest_reviews = review_results
        st.write(f"Completed iteration {iteration + 1}")

    return {
        "all_iterations": all_iterations,
        "moderation": None,
        "success": True
    }

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
4. Final Assessment including scores and recommendation
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
        
        model_type = st.selectbox("Select Model", ["GPT-4o", "Gemini"])
        review_type = st.selectbox("Select Review Type", ["Paper", "Grant", "Poster"])
        num_reviewers = st.number_input("Number of Reviewers", 1, 10, 2)
        num_iterations = st.number_input("Discussion Iterations", 1, 10, 2)
        use_moderator = st.checkbox("Include Moderator", value=True) if num_reviewers > 1 else False
        
        expertises = []
        custom_prompts = []
        
        with st.expander("Configure Reviewers"):
            for i in range(num_reviewers):
                col1, col2 = st.columns(2)
                with col1:
                    expertise = st.text_input(f"Expertise {i+1}", f"Expert {i+1}")
                    expertises.append(expertise)
                with col2:
                    prompt = st.text_area(f"Prompt {i+1}", get_default_prompt(review_type, expertise))
                    custom_prompts.append(prompt)
        
        uploaded_file = st.file_uploader(f"Upload {review_type} (PDF)", type=["pdf"])
        
        if uploaded_file and st.button("Start Review"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                content = extract_pdf_content(uploaded_file)[0]
                agents = create_review_agents(
                    num_reviewers, 
                    review_type.lower(), 
                    use_moderator,
                    model_type
                )
                
                results = process_reviews_with_debate(
                    content=content,
                    agents=agents,
                    expertises=expertises,
                    custom_prompts=custom_prompts,
                    review_type=review_type.lower(),
                    num_iterations=num_iterations,
                    model_type=model_type,
                    progress_callback=lambda p, s: (progress_bar.progress(int(p)), status_text.text(s))
                )
                
                progress_bar.empty()
                status_text.empty()
                st.success("Review completed!")
                
                # Display reviews
                for iteration_idx, iteration_reviews in enumerate(results["all_iterations"]):
                    st.subheader(f"Iteration {iteration_idx + 1}")
                    
                    for review in iteration_reviews:
                        with st.expander(f"Review by {review['expertise']}", expanded=True):
                            if review.get("success", False):
                                st.markdown(review["review"])
                            else:
                                st.error(review["review"])
                
                # Display moderator analysis if available
                if results.get("moderation"):
                    st.subheader("Final Moderator Analysis")
                    st.markdown(results["moderation"])
                        
            except Exception as e:
                st.error(f"Review process error: {str(e)}")
                logging.exception("Error in review process:")
                if st.checkbox("Debug Mode"):
                    st.exception(e)
                    
    except Exception as e:
        st.error(f"Page initialization error: {str(e)}")
        logging.exception("Error in scientific_review_page:")

if __name__ == "__main__":
    try:
        scientific_review_page()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logging.exception("Error in main:")
