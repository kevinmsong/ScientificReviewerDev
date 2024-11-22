import streamlit as stimport streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import fitz
import json
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Configure logging and OpenAI client
logging.basicConfig(level=logging.INFO)
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)

class ReviewManager:
    def __init__(self):
        self.model = "gpt-4o"
        
    def process_review(self, content: str, num_reviewers: int, doc_type: str) -> Dict[str, Any]:
        """Process document review with multiple reviewers."""
        reviews = []
        
        for i in range(num_reviewers):
            try:
                agent = ChatOpenAI(temperature=0.1, openai_api_key=api_key, model=self.model)
                prompt = self._get_review_prompt(doc_type, f"Reviewer {i+1}")
                response = agent.invoke([HumanMessage(content=f"{prompt}\n\nContent:\n{content}")])
                
                reviews.append({
                    'reviewer': f"Reviewer {i+1}",
                    'content': response.content,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logging.error(f"Error in review process: {e}")
                
        return {
            'reviews': reviews,
            'timestamp': datetime.now().isoformat(),
            'doc_type': doc_type
        }
    
    def _get_review_prompt(self, doc_type: str, reviewer: str) -> str:
        """Get review prompt based on document type."""
        return f"""As {reviewer}, please review this {doc_type} using the following structure:

1. ANALYSIS:
   - Key strengths
   - Areas for improvement
   - Technical accuracy

2. SCORING:
   Rate each category from 1-5 stars (★):
   - Scientific Merit
   - Technical Quality
   - Presentation
   - Impact

3. RECOMMENDATIONS:
   - Required changes: [REQUIRED]
   - Optional improvements: [OPTIONAL]
"""

def extract_pdf_content(pdf_file) -> str:
    """Extract text content from PDF file."""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text_content = ""
        
        for page in pdf_document:
            text_content += page.get_text()
            
        return text_content.strip()
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def display_review_results(results: Dict[str, Any]):
    """Display review results in organized format."""
    st.markdown("## Review Results")
    
    for review in results['reviews']:
        with st.expander(f"Review by {review['reviewer']}", expanded=True):
            st.markdown(review['content'])
            st.markdown(f"*Reviewed at: {review['timestamp']}*")

def main():
    st.title("Scientific Review System")
    
    # Sidebar configuration
    with st.sidebar:
        doc_type = st.selectbox(
            "Document Type",
            ["Research Paper", "Grant Proposal", "Technical Report"]
        )
        
        num_reviewers = st.slider(
            "Number of Reviewers","Number of Reviewers",
            min_value=1,
            max_value=5,
            value=2
        )
    
    # Main content area
    uploaded_file = st.file_uploader("Upload Document (PDF)", type=["pdf"])
    
    if uploaded_file:
        try:
            # Extract content
            with st.spinner("Extracting document content..."):
                content = extract_pdf_content(uploaded_file)
            
            # Process review
            if st.button("Generate Review"):
                with st.spinner("Generating reviews..."):
                    review_manager = ReviewManager()
                    results = review_manager.process_review(
                        content=content,
                        num_reviewers=num_reviewers,
                        doc_type=doc_type
                    )
                    
                    display_review_results(results)
                    
                    # Download button for results
                    st.download_button(
                        label="Download Review Report",
                        data=json.dumps(results, indent=2),
                        file_name=f"review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
