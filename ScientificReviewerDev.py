import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import fitz
import io
from PIL import Image
import base64
from typing import List, Dict, Any, Tuple, Union
import tiktoken
import time
import json
import os
from datetime import datetime
import uuid
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)

class ReviewPersistenceManager:
    def __init__(self, storage_dir: str = "data/reviews"):
        """Initialize the review persistence manager."""
        self.storage_dir = storage_dir
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self):
        """Create storage directories if they don't exist."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
    
    def _get_review_path(self, review_id: str) -> str:
        """Get the file path for a specific review."""
        return os.path.join(self.storage_dir, f"review_{review_id}.json")
    
    def save_review_session(self, review_data: Dict[str, Any]) -> Optional[str]:
        """Save a complete review session including all iterations and moderator feedback."""
        try:
            review_id = str(uuid.uuid4())
            review_data['review_id'] = review_id
            review_data['created_at'] = datetime.now().isoformat()
            
            file_path = self._get_review_path(review_id)
            with open(file_path, 'w') as f:
                json.dump(review_data, f, indent=2)
            
            return review_id
        except Exception as e:
            logging.error(f"Error saving review session: {e}")
            return None
    
    def get_review_session(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a complete review session by ID."""
        try:
            file_path = self._get_review_path(review_id)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logging.error(f"Error retrieving review session: {e}")
            return None
    
    def get_all_reviews(self, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve all review sessions with optional pagination."""
        try:
            reviews = []
            files = os.listdir(self.storage_dir)
            review_files = [f for f in files if f.startswith('review_')]
            
            # Sort by creation time (newest first)
            review_files.sort(key=lambda x: os.path.getctime(
                os.path.join(self.storage_dir, x)), reverse=True)
            
            if limit is not None:
                review_files = review_files[offset:offset + limit]
            
            for file_name in review_files:
                with open(os.path.join(self.storage_dir, file_name), 'r') as f:
                    review = json.load(f)
                    reviews.append(review)
            
            return reviews
        except Exception as e:
            logging.error(f"Error retrieving all reviews: {e}")
            return []

class ReviewContext:
    def __init__(self, storage_dir: str = "data/reviews"):
        """Initialize the review context manager."""
        self.storage_dir = storage_dir
        self.persistence_manager = ReviewPersistenceManager(storage_dir)
        
    def get_historical_context(self, expertise: str, review_type: str, limit: int = 5) -> str:
        """Get relevant historical context for a given expertise and review type."""
        try:
            all_reviews = self.persistence_manager.get_all_reviews(limit=limit)
            relevant_reviews = []
            
            for review in all_reviews:
                if review['review_type'].lower() == review_type.lower():
                    for iteration in review.get('iterations', []):
                        for rev in iteration.get('reviews', []):
                            if rev.get('expertise') == expertise and rev.get('success', False):
                                relevant_reviews.append({
                                    'review_text': rev.get('review_text', ''),
                                    'timestamp': rev.get('timestamp', ''),
                                    'scores': self._extract_scores(rev.get('review_text', ''))
                                })
            
            if relevant_reviews:
                context = f"Historical Context from Previous {review_type} Reviews:\n\n"
                for i, rev in enumerate(relevant_reviews, 1):
                    context += f"Previous Review {i} ({rev['timestamp'][:10]}):\n"
                    context += f"Key Points: {self._extract_key_points(rev['review_text'])}\n"
                    if rev['scores']:
                        context += f"Previous Scores: {rev['scores']}\n"
                    context += "---\n"
                return context
            return ""
        except Exception as e:
            logging.error(f"Error getting historical context: {e}")
            return ""
    
    def _extract_scores(self, review_text: str) -> Dict[str, float]:
        """Extract numerical scores from review text."""
        scores = {}
        try:
            score_patterns = [
                r'(\w+)\s*(?:score|rating):\s*(\d+(?:\.\d+)?)/?\d*',
                r'(\w+):\s*(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+',
            ]
            
            for pattern in score_patterns:
                matches = re.finditer(pattern, review_text, re.IGNORECASE)
                for match in matches:
                    category = match.group(1).lower()
                    score = float(match.group(2))
                    scores[category] = score
        except Exception as e:
            logging.error(f"Error extracting scores: {e}")
        return scores
    
    def _extract_key_points(self, review_text: str, max_points: int = 3) -> List[str]:
        """Extract key points from review text."""
        try:
            sections = ['strengths', 'weaknesses', 'key points', 'main findings']
            key_points = []
            
            for section in sections:
                pattern = rf"{section}:?(.*?)(?:\n\n|\Z)"
                matches = re.finditer(pattern, review_text, re.IGNORECASE)
                for match in matches:
                    points = match.group(1).strip().split('\n')
                    points = [p.strip('- ') for p in points if p.strip()]
                    key_points.extend(points[:max_points])
                    
            return key_points[:max_points]
        except Exception as e:
            logging.error(f"Error extracting key points: {e}")
            return []

def initialize_review_settings():
    """Initialize default review settings based on document type."""
    return {
        "Paper": {"reviewers": 4, "iterations": 1, "rating": "stars"},
        "Grant Proposal": {"reviewers": 3, "iterations": 2, "rating": "nih"},
        "Poster": {"reviewers": 1, "iterations": 1, "rating": "stars"}
    }

def create_review_agents(num_agents: int, review_type: str = "paper", include_moderator: bool = False) -> List[ChatOpenAI]:
    """Create review agents including a moderator if specified."""
    model = "gpt-4o"
    
    agents = [ChatOpenAI(temperature=0.1, openai_api_key=api_key, model=model) 
             for _ in range(num_agents)]
    
    if include_moderator and num_agents > 1:
        moderator_agent = ChatOpenAI(temperature=0.1, openai_api_key=api_key, model=model)
        agents.append(moderator_agent)
    
    return agents

def create_review_prompt(expertise: str, doc_type: str, venue: str, bias: int, 
                        rating_system: str, is_nih_grant: bool = False) -> str:
    """Generate a review prompt based on parameters."""
    prompt = f"""As an expert in {expertise}, you are evaluating a {doc_type} for {venue}.

{_get_bias_instruction(bias)}

Please structure your review as follows:"""

    if is_nih_grant:
        prompt += """

1. PRELIMINARY SCORES (1-9, where 1 is best):
   - Significance: [Score] - [Brief justification]
   - Innovation: [Score] - [Brief justification]
   - Approach: [Score] - [Brief justification]
   - Overall Impact: [Score] - [Brief justification]

2. DETAILED EVALUATION:"""
    
    if doc_type.lower() == "paper":
        prompt += """

1. SECTION-BY-SECTION ANALYSIS:
   For each section, provide:
   - Summary of content
   - Specific suggestions for improvement
   - Required vs. optional changes
   
2. OVERALL ASSESSMENT:
   - Strengths
   - Weaknesses
   - Detailed recommendations for revision

3. FINAL RATING:"""
        if rating_system == "stars":
            prompt += """
   Provide a 1-5 star rating where:
   ★☆☆☆☆ - Junk
   ★★☆☆☆ - Very poor
   ★★★☆☆ - Barely acceptable
   ★★★★☆ - Good
   ★★★★★ - Outstanding"""
    else:
        prompt += "\n\nFINAL RATING:"
        if rating_system == "stars":
            prompt += " (1-5 stars)"
        else:
            prompt += """
Using NIH percentile scale:
1 - Top 5% (Outstanding)
2 - Top 10% (Excellent)
3 - Top 20% (Very Good)
4 - Top 33% (Good)
5 - 50th percentile (Average)
6 - Bottom 33%
7 - Bottom 20%
8 - Bottom 10%
9 - Bottom 5%"""

    return prompt

def _get_bias_instruction(bias: int) -> str:
    """Get review bias instruction based on bias parameter."""
    bias_instructions = {
        -2: "Approach this review with a highly critical and skeptical mindset. Focus intensely on identifying flaws and shortcomings.",
        -1: "Take a somewhat critical perspective, emphasizing areas for improvement while acknowledging strengths.",
        0: "Maintain complete objectivity in your evaluation, giving equal attention to strengths and weaknesses.",
        1: "Approach the review with an encouraging mindset, emphasizing potential while noting necessary improvements.",
        2: "Take an enthusiastic and constructive approach, focusing on possibilities and positive aspects while addressing critical issues."
    }
    return bias_instructions.get(bias, bias_instructions[0])

def process_reviews_with_context(content: str, agents: List[ChatOpenAI], expertises: List[str], 
                               custom_prompts: List[str], review_type: str, 
                               num_iterations: int, progress_callback=None) -> Dict[str, Any]:
    """Process reviews with multiple iterations of debate between reviewers with persistence."""
    context_manager = ReviewContext()
    review_manager = ReviewPersistenceManager()
    
    review_session = {
        'review_type': review_type,
        'expertises': expertises,
        'num_iterations': num_iterations,
        'iterations': [],
        'metadata': {
            'started_at': datetime.now().isoformat(),
            'status': 'in_progress'
        }
    }
    
    review_id = review_manager.save_review_session(review_session)
    
    # Initialize containers for display
    review_containers = {}
    all_iterations = []
    latest_reviews = []
    
    # Process each iteration
    for iteration in range(num_iterations):
        iteration_data = {
            'iteration_number': iteration + 1,
            'reviews': [],
            'started_at': datetime.now().isoformat()
        }
        
        if progress_callback:
            progress = (iteration / num_iterations) * 100
            progress_callback(progress, f"Processing iteration {iteration + 1}/{num_iterations}")
        
        # Process each reviewer
        for i, (agent, expertise, base_prompt) in enumerate(zip(agents[:-1], expertises, custom_prompts)):
            try:
                historical_context = context_manager.get_historical_context(
                    expertise, review_type
                )
                
                debate_prompt = get_enhanced_debate_prompt(
                    expertise, iteration + 1, latest_reviews, 
                    review_type, historical_context
                )
                full_prompt = f"{base_prompt}\n\n{debate_prompt}"
                
                review_text = process_chunks_with_debate(
                    content_chunks=chunk_content(content),
                    agent=agent,
                    expertise=expertise,
                    prompt=full_prompt,
                    iteration=iteration + 1
                )
                
                review_data = {
                    'expertise': expertise,
                    'review_text': review_text,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
                
                iteration_data['reviews'].append(review_data)
                latest_reviews.append(review_data)
                
                review_manager.update_review_session(review_id, {
                    'iterations': all_iterations + [iteration_data],
                    'metadata': {
                        'last_updated': datetime.now().isoformat(),
                        'current_iteration': iteration + 1
                    }
                })
                
            except Exception as e:
                logging.error(f"Error in review process for {expertise}: {e}")
                review_data = {
                    'expertise': expertise,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'success': False
                }
                iteration_data['reviews'].append(review_data)
        
        all_iterations.append(iteration_data)
    
    # Process moderator analysis if applicable
    moderation_result = None
    if len(agents) > len(expertises):
        try:
            historical_moderation = context_manager.get_historical_context(
                "moderator", review_type
            )
            
            moderator_prompt = create_moderator_prompt(
                all_iterations, historical_moderation
            )
            
            moderator_response = agents[-1].invoke([HumanMessage(content=moderator_prompt)])
            moderation_result = extract_content(moderator_response, "[Error in moderation]")
            
            review_manager.update_review_session(review_id, {
                'moderator_analysis': {
                    'analysis': moderation_result,
                    'timestamp': datetime.now().isoformat()
                },
                'metadata': {
                    'status': 'completed',
                    'completed_at': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logging.error(f"Error in moderation process: {e}")
            moderation_result = f"Error in moderation: {str(e)}"
    
    return {
        'review_id': review_id,
        'iterations': all_iterations,
        'moderation': moderation_result
    }

def scientific_review_page():
    st.header("Multi-Agent Scientific Review System")
    
    # Initialize default settings
    review_defaults = initialize_review_settings()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Review Configuration")
        
        # Reviewer bias slider
        bias = st.slider(
            "Reviewer Bias",
            min_value=-2,
            max_value=2,
            value=0,
            help="""
            -2: Extremely negative and biased
            -1: Somewhat negative and biased
            0: Unbiased/objective
            1: Positive and enthusiastic
            2: Extremely positive and passionate
            """
        )
        
        # Additional settings
        st.markdown("---")
        st.subheader("Advanced Settings")
        
        # Model temperature
        temperature = st.slider(
            "Model Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controls randomness in model responses"
        )
        
        # Debug mode
        debug_mode = st.checkbox(
            "Debug Mode",
            value=False,
            help="Show detailed logging information"
        )
    
    # Main configuration area
    st.subheader("Document Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        # Document type selection
        doc_type = st.selectbox(
            "Document Type",
            options=list(review_defaults.keys()),
            key="doc_type"
        )
        
        # Set default values based on document type
        default_settings = review_defaults[doc_type]
        
        # Dissemination venue
        venue = st.text_input(
            "Dissemination Venue",
            placeholder="e.g., Nature, NIH R01, Conference Name",
            help="Where this work is intended to be published/presented"
        )
    
    with col2:
        # Rating system selection
        rating_system = st.radio(
            "Rating System",
            options=["stars", "nih"],
            format_func=lambda x: "Star Rating (1-5)" if x == "stars" else "NIH Scale (1-9)",
            horizontal=True
        )
        
        # NIH grant specific options
        is_nih_grant = st.checkbox(
            "NIH Grant Review Format",
            value=True if doc_type == "Grant Proposal" else False,
            help="Include separate scores for Significance, Innovation, and Approach"
        )
    
    # Reviewer configuration
    st.subheader("Reviewer Configuration")
    
    # Initialize session state for reviewers if not exists
    if 'num_reviewers' not in st.session_state:
        st.session_state.num_reviewers = default_settings['reviewers']
    
    # Add/remove reviewer buttons
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Add Reviewer"):
            st.session_state.num_reviewers += 1
        if st.button("Remove Reviewer") and st.session_state.num_reviewers > 1:
            st.session_state.num_reviewers -= 1
    
    # Reviewer expertise inputs
    expertises = []
    for i in range(st.session_state.num_reviewers):
        expertise = st.text_input(
            f"Reviewer {i+1} Expertise",
            value=f"Scientific Expert {i+1}",
            key=f"expertise_{i}"
        )
        expertises.append(expertise)
    
    # Number of iterations
    num_iterations = st.number_input(
        "Number of Discussion Iterations",
        min_value=1,
        max_value=5,
        value=default_settings['iterations'],
        help="Number of rounds of discussion between reviewers"
    )
    
    # File upload and processing
    uploaded_file = st.file_uploader(
        f"Upload {doc_type} (PDF)",
        type=["pdf"],
        key="uploaded_file"
    )
    
    if uploaded_file and st.button("Start Review", disabled=not venue):
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, status):
                progress_bar.progress(int(progress))
                status_text.text(status)
            
            # Extract content
            update_progress(10, "Extracting content from PDF...")
            content = extract_pdf_content(uploaded_file)[0]
            
            # Create agents
            update_progress(20, "Initializing review agents...")
            agents = create_review_agents(
                num_reviewers=len(expertises),
                review_type=doc_type.lower(),
                include_moderator=len(expertises) > 1
            )
            
            # Generate custom prompts
            custom_prompts = [
                create_review_prompt(
                    expertise=exp,
                    doc_type=doc_type,
                    venue=venue,
                    bias=bias,
                    rating_system=rating_system,
                    is_nih_grant=is_nih_grant
                ) for exp in expertises
            ]
            
            # Process reviews
            update_progress(30, "Starting review process...")
            results = process_reviews_with_context(
                content=content,
                agents=agents,
                expertises=expertises,
                custom_prompts=custom_prompts,
                review_type=doc_type.lower(),
                num_iterations=num_iterations,
                progress_callback=update_progress
            )
            
            update_progress(100, "Review process completed!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.success("Review process completed successfully!")
            display_review_results_with_debate(results)
            
        except Exception as e:
            st.error(f"An error occurred during the review process: {str(e)}")
            logging.exception("Error in review process:")
            
            if debug_mode:
                st.exception(e)
            
            st.warning("Please try again or check your inputs.")

def display_review_results_with_debate(results: Dict[str, Any]) -> None:
    """Display results from iterative review process with enhanced formatting."""
    try:
        # Display iterations
        for iteration_idx, iteration_data in enumerate(results["iterations"], 1):
            with st.expander(f"Iteration {iteration_idx}", expanded=True):
                st.markdown(f"### Iteration {iteration_idx}")
                
                # Display each review in this iteration
                for review in iteration_data["reviews"]:
                    if review.get("success", False):
                        with st.expander(f"Review by {review['expertise']}", expanded=True):
                            # Parse and display scores separately if present
                            scores = extract_scores_from_review(review['review_text'])
                            if scores:
                                st.markdown("#### Scores")
                                for category, score in scores.items():
                                    st.markdown(f"**{category}**: {score}")
                                st.markdown("---")
                            
                            # Display main review content
                            sections = review['review_text'].split('\n\n')
                            for section in sections:
                                if section.strip():
                                    st.markdown(section.strip())
                            
                            # Display timestamp
                            st.markdown(f"*Reviewed at: {review['timestamp']}*")
                    else:
                        st.error(f"Error in review by {review['expertise']}: {review.get('error', 'Unknown error')}")
        
        # Display final moderation
        if results.get("moderation"):
            with st.expander("Final Moderator Analysis", expanded=True):
                st.markdown("### Final Moderator Analysis")
                if not str(results["moderation"]).startswith("[Error"):
                    sections = results["moderation"].split('\n\n')
                    for section in sections:
                        if section.strip():
                            st.markdown(section.strip())
                            st.markdown("---")
                else:
                    st.error(results["moderation"])
    
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        logging.exception("Error in display_review_results_with_debate:")

def extract_scores_from_review(review_text: str) -> Dict[str, Any]:
    """Extract scores from review text."""
    scores = {}
    try:
        # Look for various score formats
        patterns = [
            r'(\w+)\s*(?:score|rating):\s*(\d+(?:\.\d+)?)',
            r'(\w+):\s*(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+',
            r'(\w+):\s*([★]+(?:☆)*)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, review_text, re.IGNORECASE)
            for match in matches:
                category = match.group(1).strip()
                score = match.group(2)
                # Convert star ratings to numerical scores
                if '★' in score:
                    score = len(score.replace('☆', ''))
                else:
                    score = float(score)
                scores[category] = score
                
    except Exception as e:
        logging.error(f"Error extracting scores: {e}")
    
    return scores

def main():
    st.set_page_config(
        page_title="Multi-Agent Scientific Review System",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .stExpander {
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
        }
        .stTextArea>div>div>textarea {
            font-family: monospace;
        }
        .review-section {
            margin: 1rem 0;
            padding: 1rem;
            border-left: 3px solid #4CAF50;border-left: 3px solid #4CAF50;border-left: 3px solid #4CAF50;border-left: 3px solid #4CAF50;
            background-color: #f8f9fa;
        }
        .review-header {
            font-weight: bold;
            color: #2C3E50;
            margin-bottom: 0.5rem;
        }
        .score-display {
            font-weight: bold;
            color: #E74C3C;
        }
        </style>
        """, unsafe_allow_html=True)
    
    scientific_review_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Unexpected error in main application:")
