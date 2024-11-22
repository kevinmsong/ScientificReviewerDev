import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import fitz
import json
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Configure logging and OpenAI client
logging.basicConfig(level=logging.INFO)
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)

# Pre-configured expertise options
EXPERTISE_OPTIONS = {
    "Technical": [
        "Data Scientist",
        "Machine Learning Engineer",
        "Software Engineer",
        "Systems Architect",
        "Database Expert"
    ],
    "Scientific": [
        "Research Scientist",
        "Bioinformatician",
        "Statistical Expert",
        "Computational Biologist",
        "Quantitative Analyst"
    ],
    "Domain": [
        "Subject Matter Expert",
        "Industry Specialist",
        "Field Researcher",
        "Clinical Expert",
        "Domain Consultant"
    ]
}

# Review settings
REVIEW_DEFAULTS = {
    "Paper": {
        "reviewers": 4,
        "iterations": 1,
        "scoring": "stars",
        "scale": "1-5 stars"
    },
    "Grant": {
        "reviewers": 3,
        "iterations": 2,
        "scoring": "nih",
        "scale": "1-9 NIH scale"
    }
}

class ReviewManager:
    def __init__(self):
        self.model = "gpt-4o"
        
    def process_review(self, content: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process document review based on configuration."""
        reviews = []
        iterations = []
        
        for iteration in range(config['iterations']):
            iteration_reviews = []
            
            for reviewer in config['reviewers']:
                try:
                    agent = ChatOpenAI(
                        temperature=0.1 + config.get('bias', 0) * 0.1,
                        openai_api_key=api_key,
                        model=self.model
                    )
                    
                    prompt = self._create_review_prompt(
                        doc_type=config['doc_type'],
                        expertise=reviewer['expertise'],
                        scoring_type=config['scoring'],
                        iteration=iteration + 1
                    )
                    
                    response = agent.invoke([HumanMessage(content=f"{prompt}\n\nContent:\n{content}")])
                    
                    iteration_reviews.append({
                        'reviewer': reviewer['expertise'],
                        'content': response.content,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    logging.error(f"Error in review process: {e}")
            
            iterations.append({
                'iteration_number': iteration + 1,
                'reviews': iteration_reviews
            })
        
        return {
            'iterations': iterations,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'doc_type': config['doc_type']
        }
    
    def _create_review_prompt(self, doc_type: str, expertise: str, scoring_type: str, iteration: int) -> str:
        """Create specialized review prompt based on document type and scoring system."""
        base_prompt = f"""As a {expertise}, please review this {doc_type} for iteration {iteration}.

Use the following structure:"""

        if doc_type.lower() == "grant":
            return base_prompt + """

1. SIGNIFICANCE EVALUATION:
   - Impact potential
   - Field advancement
   - Score (1-9 NIH scale)

2. INNOVATION ASSESSMENT:
   - Novel aspects
   - Technical advances
   - Score (1-9 NIH scale)

3. APPROACH ANALYSIS:
   - Methodology
   - Feasibility
   - Score (1-9 NIH scale)

4. OVERALL IMPACT SCORE (1-9 NIH scale)
   Where: 1 = Exceptional, 9 = Poor

5. DETAILED RECOMMENDATIONS:
   [REQUIRED] Critical changes needed
   [OPTIONAL] Suggested improvements"""
        
        else:
            return base_prompt + """

1. SECTION-BY-SECTION ANALYSIS:
   For each major section:
   - Content summary
   - [REQUIRED] Critical changes
   - [OPTIONAL] Suggested improvements
   - Specific line edits

2. SCORING (★☆):
   Rate each category (1-5 stars):
   - Scientific Merit
   - Technical Quality
   - Presentation
   - Impact

3. RECOMMENDATIONS:
   [REQUIRED] Critical changes needed
   [OPTIONAL] Suggested improvements"""

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
    
    for iteration in results['iterations']:
        st.markdown(f"### Iteration {iteration['iteration_number']}")
        
        for review in iteration['reviews']:
            with st.expander(f"Review by {review['reviewer']}", expanded=True):
                st.markdown(review['content'])
                st.markdown(f"*Reviewed at: {review['timestamp']}*")

def main():
    st.title("Scientific Review System")
    
    # Main configuration area
    st.markdown("## Document Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        doc_type = st.selectbox(
            "Document Type",
            list(REVIEW_DEFAULTS.keys())
        )
        
        venue = st.text_input(
            "Dissemination Venue",
            placeholder="e.g., Nature, NIH R01, Conference Name"
        )
    
    with col2:
        scoring_system = st.radio(
            "Scoring System",
            ["stars", "nih"],
            format_func=lambda x: "Star Rating (1-5)" if x == "stars" else "NIH Scale (1-9)"
        )
    
    # Reviewer configuration
    st.markdown("## Reviewer Configuration")
    default_settings = REVIEW_DEFAULTS[doc_type]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_reviewers = st.number_input(
            "Number of Reviewers",
            min_value=1,
            max_value=10,
            value=default_settings['reviewers']
        )
    
    with col2:
        num_iterations = st.number_input(
            "Number of Iterations",
            min_value=1,
            max_value=10,
            value=default_settings['iterations']
        )
    
    with col3:
        reviewer_bias = st.select_slider(
            "Reviewer Bias",
            options=[-2, -1, 0, 1, 2],
            value=0,
            format_func=lambda x: {
                -2: "Extremely Negative",
                -1: "Somewhat Negative",
                0: "Unbiased",
                1: "Somewhat Positive",
                2: "Extremely Positive"
            }[x]
        )
    
    # Reviewer expertise selection
    reviewers = []
    for i in range(num_reviewers):
        st.markdown(f"### Reviewer {i+1}")
        col1, col2 = st.columns(2)
        
        with col1:
            expertise_category = st.selectbox(
                "Expertise Category",
                list(EXPERTISE_OPTIONS.keys()),
                key=f"cat_{i}"
            )
        
        with col2:
            expertise = st.selectbox(
                "Specific Expertise",
                EXPERTISE_OPTIONS[expertise_category],
                key=f"exp_{i}"
            )
        
        reviewers.append({"expertise": expertise})
    
    # Document upload
    st.markdown("## Document Upload")
    uploaded_file = st.file_uploader("Upload Document (PDF)", type=["pdf"])
    
    if uploaded_file:
        try:
            # Extract content
            with st.spinner("Extracting document content..."):
                content = extract_pdf_content(uploaded_file)
            
            # Process review
            if st.button("Generate Review", type="primary"):
                config = {
                    "doc_type": doc_type,
                    "venue": venue,
                    "scoring": scoring_system,
                    "reviewers": reviewers,
                    "iterations": num_iterations,
                    "bias": reviewer_bias
                }
                
                with st.spinner("Generating reviews..."):
                    review_manager = ReviewManager()
                    results = review_manager.process_review(
                        content=content,
                        config=config
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
