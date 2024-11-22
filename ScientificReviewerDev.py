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
    "Computational Drug Discovery": [
        "Molecular Modeling Expert",
        "Drug Design Specialist",
        "Cheminformatics Scientist",
        "Computational Chemist",
        "Structure-Based Drug Design Expert",
        "Virtual Screening Specialist",
        "QSAR Modeling Expert",
        "Molecular Dynamics Specialist",
        "Fragment-Based Design Expert",
        "Quantum Chemistry Specialist"
    ],
    "Systems Pharmacology": [
        "Systems Pharmacologist",
        "PK/PD Modeling Expert",
        "Network Pharmacology Specialist",
        "Multi-Scale Modeling Expert",
        "Biological Network Analyst",
        "Quantitative Systems Pharmacologist",
        "Drug-Response Modeling Expert",
        "Clinical Pharmacology Specialist",
        "Systems Biology Expert",
        "Pathway Analysis Specialist"
    ],
    "AI/ML in Drug Discovery": [
        "AI Drug Discovery Scientist",
        "Deep Learning Specialist",
        "Generative AI Expert",
        "Machine Learning Engineer",
        "AI Model Validation Expert",
        "Biomedical AI Researcher",
        "Drug Response AI Specialist",
        "AI-Driven Lead Optimization Expert",
        "Computer Vision in Drug Discovery",
        "NLP in Drug Discovery Expert"
    ],
    "Technical Specialties": [
        "Data Scientist",
        "Software Engineer",
        "Systems Architect",
        "Database Expert",
        "Cloud Computing Specialist",
        "High-Performance Computing Expert",
        "DevOps Engineer",
        "Data Engineer",
        "Research Software Engineer",
        "Bioinformatics Engineer"
    ],
    "Biological Sciences": [
        "Molecular Biologist",
        "Biochemist",
        "Pharmacologist",
        "Structural Biologist",
        "Cell Biologist",
        "Geneticist",
        "Immunologist",
        "Protein Scientist",
        "Medicinal Chemist",
        "Toxicologist"
    ],
    "Clinical & Translational": [
        "Clinical Pharmacologist",
        "Translational Scientist",
        "Clinical Trial Designer",
        "Regulatory Affairs Expert",
        "Drug Safety Specialist",
        "Biomarker Specialist",
        "Clinical Data Scientist",
        "Patient Stratification Expert",
        "Precision Medicine Specialist",
        "Drug Development Expert"
    ],
    "Domain Integration": [
        "Multi-Omics Integration Expert",
        "Systems Medicine Specialist",
        "Translational Informatics Expert",
        "Drug Repurposing Specialist",
        "Target Discovery Expert",
        "Drug Resistance Analyst",
        "Precision Oncology Specialist",
        "Disease Modeling Expert",
        "Biomedical Knowledge Graph Specialist",
        "Clinical AI Integration Expert"
    ]
}

# Review settings can remain the same
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

class ModeratorAgent:
    def __init__(self, model="gpt-4o"):
        self.model = ChatOpenAI(
            temperature=0.1,
            openai_api_key=st.secrets["openai_api_key"],
            model=model
        )
    
    def summarize_discussion(self, iterations: List[Dict[str, Any]]) -> str:
        """Analyze the evolution of discussion across iterations."""
        discussion_summary = "Discussion Evolution Summary:\n\n"
        for iteration in iterations:
            discussion_summary += f"Iteration {iteration['iteration_number']} reviews:\n"
            for review in iteration['reviews']:
                discussion_summary += f"- {review['reviewer']}: {review['content']}\n\n"
        
        moderator_prompt = f"""As a moderator, analyze how the scientific discussion evolved across these iterations:

{discussion_summary}

Please provide:
1. KEY POINTS OF AGREEMENT:
   - Areas where reviewers reached consensus
   - Shared concerns or praise

2. POINTS OF CONTENTION:
   - Areas of disagreement
   - Differing perspectives

3. DISCUSSION EVOLUTION:
   - How viewpoints changed across iterations
   - How reviewers responded to each other

4. FINAL SYNTHESIS:
   - Overall consensus
   - Key recommendations
   - Final decision recommendation

Format your response using these exact sections."""

        try:
            response = self.model.invoke([HumanMessage(content=moderator_prompt)])
            return response.content
        except Exception as e:
            logging.error(f"Error in moderation: {e}")
            return f"Error generating moderation summary: {str(e)}"

class ReviewManager:
    def __init__(self):
        self.model = "gpt-4o"
        self.moderator = ModeratorAgent()
        
    def process_review(self, content: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process document review with multiple iterations and moderation."""
        iterations = []
        all_reviews = []  # Keep track of all reviews for context
        
        for iteration in range(config['iterations']):
            iteration_reviews = []
            
            # Create context from previous reviews
            previous_context = ""
            if all_reviews:
                previous_context = "\n\nPrevious reviews:\n" + \
                    "\n".join([f"Reviewer {r['reviewer']}: {r['content']}" for r in all_reviews])
            
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
                        iteration=iteration + 1,
                        previous_context=previous_context
                    )
                    
                    response = agent.invoke([HumanMessage(content=f"{prompt}\n\nContent:\n{content}")])
                    review = {
                        'reviewer': reviewer['expertise'],
                        'content': response.content,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    iteration_reviews.append(review)
                    all_reviews.append(review)
                    
                except Exception as e:
                    logging.error(f"Error in review process: {e}")
            
            iterations.append({
                'iteration_number': iteration + 1,
                'reviews': iteration_reviews
            })
        
        # Generate moderator summary
        moderator_summary = self.moderator.summarize_discussion(iterations)
        
        return {
            'iterations': iterations,
            'moderator_summary': moderator_summary,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'doc_type': config['doc_type']
        }
    
    def _create_review_prompt(self, doc_type: str, expertise: str, scoring_type: str, 
                            iteration: int, previous_context: str) -> str:
        """Create review prompt with context from previous iterations."""
        base_prompt = f"""As a {expertise}, please review this {doc_type} for iteration {iteration}.
{previous_context}

Please consider previous reviews (if any) and respond to other reviewers' points.
Use the following structure:"""

        if doc_type.lower() == "grant":
            return base_prompt + """
1. RESPONSE TO PREVIOUS REVIEWS (if applicable):
   - Areas of agreement
   - Points of disagreement
   - Additional perspectives

2. SIGNIFICANCE EVALUATION:
   - Impact potential
   - Field advancement
   - Score (1-9 NIH scale)

3. INNOVATION ASSESSMENT:
   - Novel aspects
   - Technical advances
   - Score (1-9 NIH scale)

4. APPROACH ANALYSIS:
   - Methodology
   - Feasibility
   - Score (1-9 NIH scale)

5. OVERALL IMPACT SCORE (1-9 NIH scale)
   Where: 1 = Exceptional, 9 = Poor

6. DETAILED RECOMMENDATIONS:
   [REQUIRED] Critical changes needed
   [OPTIONAL] Suggested improvements"""
        
        else:
            return base_prompt + """
1. RESPONSE TO PREVIOUS REVIEWS (if applicable):
   - Areas of agreement
   - Points of disagreement
   - Additional perspectives

2. SECTION-BY-SECTION ANALYSIS:
   For each major section:
   - Content summary
   - [REQUIRED] Critical changes
   - [OPTIONAL] Suggested improvements
   - Specific line edits

3. SCORING (★☆):
   Rate each category (1-5 stars):
   - Scientific Merit
   - Technical Quality
   - Presentation
   - Impact

4. RECOMMENDATIONS:
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
    """Display review results with improved formatting and moderator summary."""
    st.markdown("## Review Results")
    
    # Display iterations in tabs
    if results['iterations']:
        tabs = st.tabs([f"Iteration {i['iteration_number']}" for i in results['iterations']])
        
        for tab, iteration in zip(tabs, results['iterations']):
            with tab:
                for review in iteration['reviews']:
                    with st.expander(f"Review by {review['reviewer']}", expanded=True):
                        # Parse and display structured content
                        content = review['content']
                        sections = content.split('\n\n')
                        
                        for section in sections:
                            if section.strip():
                                st.markdown(section)
                        
                        st.markdown(f"*Reviewed at: {review['timestamp']}*")
                        st.markdown("---")
    
    # Display moderator summary
    if 'moderator_summary' in results:
        with st.expander("🎯 Moderator Analysis", expanded=True):
            st.markdown(results['moderator_summary'])

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
