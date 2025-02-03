import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import fitz
import io
from PIL import Image
import base64
from typing import List, Dict, Any, Tuple, Unionfrom typing import List, Dict, Any, Tuple, Union
import tiktoken
import google.generativeai as genai
import re

logging.basicConfig(level=logging.INFO)

def create_memoryless_agents(expertises: List[Dict], include_moderator: bool = False) -> List[Union[ChatOpenAI, Any]]:
    """Create agents without memory persistence."""
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

def process_chunk_memoryless(chunk: str, agent: Union[ChatOpenAI, Any], expertise: str, prompt: str, model_type: str) -> str:
    """Process a single text chunk without maintaining memory."""
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

def process_review_memoryless(content: str, agents: List[Union[ChatOpenAI, Any]], expertises: List[Dict], custom_prompts: List[str]) -> Dict[str, Any]:
    """Process review iteratively with memoryless agents."""
    review_results = []
    
    for i, (agent, expertise, prompt) in enumerate(zip(agents, expertises, custom_prompts)):
        st.write(f"Processing review from {expertise['name']}...")
        try:
            review_text = process_chunk_memoryless(content, agent, expertise['name'], prompt, expertise['model'])
            review_results.append({
                "expertise": expertise,
                "review": review_text,
                "success": True
            })
            with st.expander(f"Review by {expertise['name']} ({expertise['model']})", expanded=True):
                st.markdown(review_text)
        except Exception as e:
            logging.error(f"Error processing agent {expertise['name']}: {str(e)}")
            review_results.append({
                "expertise": expertise,
                "review": f"Error: {str(e)}",
                "success": False
            })
    
    return {"reviews": review_results, "success": True}

def scientific_review_page():
    st.set_page_config(page_title="Scientific Reviewer", layout="wide")
    st.header("Scientific Review System")
    st.caption("v2.2.0 - Memoryless AI Mode")
    
    review_type = st.selectbox("Select Review Type", ["Paper", "Grant", "Poster"])
    num_reviewers = st.number_input("Number of Reviewers", 1, 10, 2)
    use_moderator = st.checkbox("Include Moderator", value=True) if num_reviewers > 1 else False
    
    expertises = []
    custom_prompts = []
    
    with st.expander("Configure Reviewers"):
        for i in range(num_reviewers):
            st.subheader(f"Reviewer {i+1}")
            col1, col2 = st.columns([1, 2])
            with col1:
                expertise = st.text_input(f"Expertise", value=f"Expert {i+1}", key=f"expertise_{i}")
                model_type = st.selectbox("Model", ["GPT-4o", "Gemini 2.0 Flash"], key=f"model_{i}")
            with col2:
                prompt = st.text_area("Review Guidelines", value=f"Review this {review_type.lower()} as an expert in {expertise}", key=f"prompt_{i}")
            
            expertises.append({"name": expertise, "model": model_type})
            custom_prompts.append(prompt)
    
    uploaded_file = st.file_uploader(f"Upload {review_type} (PDF)", type=["pdf"])
    
    if uploaded_file and st.button("Start Review"):
        content, _ = extract_pdf_content(uploaded_file)
        agents = create_memoryless_agents(expertises, use_moderator)
        process_review_memoryless(content, agents, expertises, custom_prompts)

if __name__ == "__main__":
    scientific_review_page()
