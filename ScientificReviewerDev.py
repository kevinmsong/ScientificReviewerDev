import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Multi-Agent Scientific Review System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import fitz
import io
from PIL import Image
import base64
from typing import List, Dict, Any, Tuple, Union, Optional
import tiktoken
import time
import json
import os
from datetime import datetime
import uuid
import re
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)
DEFAULT_MODEL = "gpt-4o"

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .reviewer-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .score-display {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .history-item {
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        transition: background-color 0.2s;
    }
    .history-item:hover {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# Add this function after the imports and before any other functions
def get_default_prompts() -> Dict[str, Dict[str, str]]:
    """Get default prompts for different document types and reviewer roles."""
    return {
        "Paper": {
            "reviewer": """You are evaluating this scientific paper with the following structured approach:

1. SECTION-BY-SECTION ANALYSIS:
   - Introduction & Background
     * Current content summary
     * Required changes
     * Optional improvements
     * Line-specific edits
   
   - Methods
     * Current content summary
     * Required changes
     * Optional improvements
     * Line-specific edits
   
   - Results
     * Current content summary
     * Required changes
     * Optional improvements
     * Line-specific edits
   
   - Discussion
     * Current content summary
     * Required changes
     * Optional improvements
     * Line-specific edits

2. OVERALL ASSESSMENT:
   - Scientific Merit
   - Technical Quality
   - Presentation
   - Impact

3. SCORING (1-5 stars):
   ★☆☆☆☆ - Junk
   ★★☆☆☆ - Very poor
   ★★★☆☆ - Barely acceptable
   ★★★★☆ - Good
   ★★★★★ - Outstanding

4. RECOMMENDATIONS:
   [REQUIRED] - Critical changes needed for acceptance
   [OPTIONAL] - Suggested improvements""",
            
            "moderator": """As a senior moderator, analyze the complete review discussion:

1. REVIEW SYNTHESIS:
   - Key agreements
   - Points of contention
   - Critical recommendations

2. DECISION FRAMEWORK:
   - Essential revisions
   - Secondary improvements
   - Publication readiness

3. FINAL RECOMMENDATION:
   - Accept/Revise/Reject
   - Required changes
   - Timeline expectations"""
        },
        
        "Grant Proposal": {
            "reviewer": """Evaluate this grant proposal using NIH criteria:

1. SIGNIFICANCE (Score 1-9):
   - Scientific premise
   - Impact potential
   - Field advancement

2. INNOVATION (Score 1-9):
   - Novel concepts
   - Methodology advances
   - Technical innovation

3. APPROACH (Score 1-9):
   - Methodology soundness
   - Timeline feasibility
   - Risk management

4. OVERALL IMPACT (Score 1-9):
   - Potential influence
   - Success likelihood
   - Resource justification""",
            
            "moderator": """Synthesize grant reviews focusing on:

1. SCIENTIFIC MERIT:
   - Impact potential
   - Innovation level
   - Methodology strength

2. FEASIBILITY:
   - Resource availability
   - Timeline realism
   - Risk assessment

3. FUNDING RECOMMENDATION:
   - Priority level
   - Required modifications
   - Support justification"""
        },
        
        "Poster": {
            "reviewer": """Assess this scientific poster considering:

1. VISUAL PRESENTATION:
   - Layout effectiveness
   - Graphics quality
   - Text readability

2. CONTENT QUALITY:
   - Scientific accuracy
   - Data presentation
   - Methods clarity

3. SCORING (1-5 stars):
   - Visual Design
   - Content Quality
   - Technical Merit""",
            
            "moderator": """Evaluate poster presentation focusing on:

1. COMMUNICATION EFFECTIVENESS:
   - Visual impact
   - Message clarity
   - Technical accuracy

2. IMPROVEMENTS:
   - Critical revisions
   - Enhancement suggestions
   - Presentation tips"""
        }
    }

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

class EnhancedReviewContext:
    def __init__(self, storage_dir: str = "data/reviews"):
        """Initialize enhanced review context manager with comprehensive history."""
        self.storage_dir = storage_dir
        self.persistence_manager = ReviewPersistenceManager(storage_dir)
        self.reviewer_history = {}
        self.load_reviewer_history()
    
    def load_reviewer_history(self):
        """Load complete reviewer history from stored reviews."""
        try:
            all_reviews = self.persistence_manager.get_all_reviews()
            for review in all_reviews:
                self._process_review_for_history(review)
        except Exception as e:
            logging.error(f"Error loading reviewer history: {e}")
    
    def _process_review_for_history(self, review: Dict[str, Any]):
        """Process a review to extract reviewer history and patterns."""
        try:
            doc_type = review.get('document_type', '').lower()
            venue = review.get('venue', '')
            
            for iteration in review.get('iterations', []):
                for rev in iteration.get('reviews', []):
                    if rev.get('success', False):
                        expertise = rev.get('expertise', '')
                        if expertise not in self.reviewer_history:
                            self.reviewer_history[expertise] = {
                                'reviews': [],
                                'patterns': {
                                    'scoring_trends': defaultdict(list),
                                    'common_critiques': defaultdict(int),
                                    'recommendation_patterns': defaultdict(int),
                                    'expertise_areas': defaultdict(int)
                                },
                                'venues': set(),
                                'doc_types': set()
                            }
                        
                        # Store review information
                        review_info = {
                            'timestamp': rev.get('timestamp', ''),
                            'doc_type': doc_type,
                            'venue': venue,
                            'review_text': rev.get('review_text', ''),
                            'scores': self._extract_scores(rev.get('review_text', '')),
                            'key_points': self._extract_key_points(rev.get('review_text', '')),
                            'critiques': self._extract_critiques(rev.get('review_text', '')),
                            'recommendations': self._extract_recommendations(rev.get('review_text', ''))
                        }
                        
                        self.reviewer_history[expertise]['reviews'].append(review_info)
                        self.reviewer_history[expertise]['venues'].add(venue)
                        self.reviewer_history[expertise]['doc_types'].add(doc_type)
                        
                        # Update patterns
                        self._update_reviewer_patterns(expertise, review_info)
        except Exception as e:
            logging.error(f"Error processing review for history: {e}")

    def _update_reviewer_patterns(self, expertise: str, review_info: Dict[str, Any]):
        """Analyze and update reviewer patterns and tendencies."""
        patterns = self.reviewer_history[expertise]['patterns']
        
        # Update scoring trends
        for category, score in review_info['scores'].items():
            patterns['scoring_trends'][category].append(score)
        
        # Update critique patterns
        for critique in review_info['critiques']:
            patterns['common_critiques'][critique.lower()] += 1
        
        # Update recommendation patterns
        for rec in review_info['recommendations']:
            patterns['recommendation_patterns'][rec.lower()] += 1
        
        # Extract and update expertise areas
        expertise_keywords = self._extract_expertise_keywords(review_info['review_text'])
        for keyword in expertise_keywords:
            patterns['expertise_areas'][keyword] += 1

    def get_reviewer_context(self, expertise: str, doc_type: str, venue: str) -> str:
        """Get comprehensive context for a reviewer."""
        if expertise not in self.reviewer_history:
            return ""
        
        history = self.reviewer_history[expertise]
        context = f"""Historical Review Context for {expertise}:

1. Experience Summary:
- Total Reviews: {len(history['reviews'])}
- Document Types: {', '.join(history['doc_types'])}
- Previous Venues: {', '.join(history['venues'])}

2. Scoring Patterns:"""
        
        # Add scoring trends
        for category, scores in history['patterns']['scoring_trends'].items():
            if scores:
                avg_score = sum(scores) / len(scores)
                context += f"\n- {category.title()}: Avg {avg_score:.1f}"
        
        # Add common critiques
        context += "\n\n3. Common Critiques:"
        top_critiques = sorted(history['patterns']['common_critiques'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        for critique, count in top_critiques:
            context += f"\n- {critique.capitalize()} ({count} occurrences)"
        
        # Add recommendation patterns
        context += "\n\n4. Common Recommendations:"
        top_recs = sorted(history['patterns']['recommendation_patterns'].items(), 
                         key=lambda x: x[1], reverse=True)[:5]
        for rec, count in top_recs:
            context += f"\n- {rec.capitalize()} ({count} occurrences)"
        
        # Add relevant previous reviews
        relevant_reviews = [r for r in history['reviews'] 
                          if r['doc_type'] == doc_type.lower() 
                          and r['venue'].lower() == venue.lower()][:3]
        
        if relevant_reviews:
            context += f"\n\n5. Recent Relevant Reviews for {venue}:"
            for idx, review in enumerate(relevant_reviews, 1):
                context += f"\n\nReview {idx} ({review['timestamp'][:10]}):"
                context += f"\nKey Points: {', '.join(review['key_points'])}"
                if review['scores']:
                    context += f"\nScores: {json.dumps(review['scores'], indent=2)}"
        
        return context
    
    def _extract_scores(self, review_text: str) -> Dict[str, float]:
        """Extract numerical scores from review text."""
        scores = {}
        try:
            score_patterns = [
                r'(\w+)\s*(?:score|rating):\s*(\d+(?:\.\d+)?)/?\d*',
                r'(\w+):\s*(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+',
                r'(\w+):\s*([★]+(?:☆)*)',
            ]
            
            for pattern in score_patterns:
                matches = re.finditer(pattern, review_text, re.IGNORECASE)
                for match in matches:
                    category = match.group(1).lower()
                    score = match.group(2)
                    if '★' in str(score):
                        score = len(score.replace('☆', ''))
                    else:
                        score = float(score)
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
    
    def _extract_critiques(self, review_text: str) -> List[str]:
        """Extract critiques and criticisms from review text."""
        critiques = []
        try:
            critique_patterns = [
                r'(?:weakness(?:es)?|criticism(?:s)?|concern(?:s)?|issue(?:s)?):?(.*?)(?:\n\n|\Z)',
                r'(?:needs?|requires?|should):?(.*?)(?:\n|$)',
                r'\[REQUIRED\](.*?)(?:\n|$)'
            ]
            
            for pattern in critique_patterns:
                matches = re.finditer(pattern, review_text, re.IGNORECASE)
                for match in matches:
                    critique = match.group(1).strip()
                    if critique:
                        critiques.append(critique)
            
            return critiques
        except Exception as e:
            logging.error(f"Error extracting critiques: {e}")
            return []
    
    def _extract_recommendations(self, review_text: str) -> List[str]:
        """Extract recommendations from review text."""
        recommendations = []
        try:
            rec_patterns = [
                r'(?:recommend(?:ation)?(?:s)?|suggest(?:ion)?(?:s)?):?(.*?)(?:\n\n|\Z)',
                r'(?:improve(?:ment)?(?:s)?):?(.*?)(?:\n\n|\Z)',
                r'\[OPTIONAL\](.*?)(?:\n|$)'
            ]
            
            for pattern in rec_patterns:
                matches = re.finditer(pattern, review_text, re.IGNORECASE)
                for match in matches:
                    rec = match.group(1).strip()
                    if rec:
                        recommendations.append(rec)
            
            return recommendations
        except Exception as e:
            logging.error(f"Error extracting recommendations: {e}")
            return []
    
    def _extract_expertise_keywords(self, review_text: str) -> List[str]:
        """Extract keywords indicating expertise areas."""
        keywords = []
        try:
            # Add your expertise keyword extraction logic here
            # This could involve NLP techniques, keyword matching, etc.
            pass
        except Exception as e:
            logging.error(f"Error extracting expertise keywords: {e}")
        return keywords
    
class ReviewProcessor:
    def __init__(self, context_manager: EnhancedReviewContext):
        """Initialize the review processor with context management."""
        self.context_manager = context_manager
        self.current_session = None
    
    def process_review(self, content: str, agents: List[ChatOpenAI], expertises: List[str],
                      custom_prompts: List[str], review_type: str, venue: str,
                      num_iterations: int, progress_callback=None) -> Dict[str, Any]:
        """Process a complete review with multiple agents and iterations."""
        try:
            self.current_session = {
                'review_type': review_type,
                'venue': venue,
                'expertises': expertises,
                'iterations': [],
                'metadata': {
                    'started_at': datetime.now().isoformat(),
                    'status': 'in_progress'
                }
            }
            
            content_chunks = self.chunk_content(content)
            all_iterations = []
            latest_reviews = []
            
            # Process each iteration
            for iteration in range(num_iterations):
                if progress_callback:
                    progress = (iteration / num_iterations) * 100
                    progress_callback(progress, f"Processing iteration {iteration + 1}/{num_iterations}")
                
                iteration_results = self.process_iteration(
                    content_chunks=content_chunks,
                    agents=agents,
                    expertises=expertises,
                    custom_prompts=custom_prompts,
                    review_type=review_type,
                    venue=venue,
                    iteration_number=iteration + 1,
                    previous_reviews=latest_reviews
                )
                
                all_iterations.append(iteration_results)
                latest_reviews = iteration_results['reviews']
            
            # Process moderator review if applicable
            moderation_result = None
            if len(agents) > len(expertises):
                moderation_result = self.process_moderation(
                    agents[-1], all_iterations, review_type, venue
                )
            
            self.current_session['iterations'] = all_iterations
            self.current_session['moderation'] = moderation_result
            self.current_session['metadata']['status'] = 'completed'
            self.current_session['metadata']['completed_at'] = datetime.now().isoformat()
            
            return self.current_session
            
        except Exception as e:
            logging.error(f"Error in review process: {e}")
            if self.current_session:
                self.current_session['metadata']['status'] = 'error'
                self.current_session['metadata']['error'] = str(e)
            raise
    
    def process_iteration(self, content_chunks: List[str], agents: List[ChatOpenAI],
                         expertises: List[str], custom_prompts: List[str],
                         review_type: str, venue: str, iteration_number: int,
                         previous_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a single iteration of reviews."""
        iteration_data = {
            'iteration_number': iteration_number,
            'reviews': [],
            'started_at': datetime.now().isoformat()
        }
        
        for i, (agent, expertise, base_prompt) in enumerate(zip(agents[:-1], expertises, custom_prompts)):
            try:
                # Get historical context for this reviewer
                reviewer_context = self.context_manager.get_reviewer_context(
                    expertise, review_type, venue
                )
                
                # Create enhanced prompt
                full_prompt = self.create_enhanced_prompt(
                    base_prompt=base_prompt,
                    expertise=expertise,
                    review_type=review_type,
                    venue=venue,
                    iteration=iteration_number,
                    previous_reviews=previous_reviews,
                    historical_context=reviewer_context
                )
                
                # Process review
                review_result = self.process_single_review(
                    agent=agent,
                    content_chunks=content_chunks,
                    prompt=full_prompt,
                    expertise=expertise
                )
                
                iteration_data['reviews'].append({
                    'expertise': expertise,
                    'review_text': review_result,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
                
            except Exception as e:
                logging.error(f"Error processing review for {expertise}: {e}")
                iteration_data['reviews'].append({
                    'expertise': expertise,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'success': False
                })
        
        iteration_data['completed_at'] = datetime.now().isoformat()
        return iteration_data
    
    def process_single_review(self, agent: ChatOpenAI, content_chunks: List[str],
                            prompt: str, expertise: str) -> str:
        """Process a single review across content chunks."""
        chunk_reviews = []
        
        for i, chunk in enumerate(content_chunks):
            chunk_prompt = f"""Reviewing part {i+1} of {len(content_chunks)}:

{prompt}

Content part {i+1}/{len(content_chunks)}:
{chunk}"""

            try:
                response = agent.invoke([HumanMessage(content=chunk_prompt)])
                review_text = self.extract_content(response)
                chunk_reviews.append(review_text)
            except Exception as e:
                logging.error(f"Error processing chunk {i+1} for {expertise}: {e}")
                chunk_reviews.append(f"[Error in chunk {i+1}: {str(e)}]")
        
        if len(content_chunks) > 1:
            # Compile chunks into final review
            compilation_prompt = f"""Please compile your reviews of all {len(content_chunks)} parts into a single coherent review.

Previous chunk reviews:
{''.join(chunk_reviews)}

Please provide a consolidated review addressing all sections of the document."""

            try:
                compilation_response = agent.invoke([HumanMessage(content=compilation_prompt)])
                return self.extract_content(compilation_response)
            except Exception as e:
                logging.error(f"Error compiling review for {expertise}: {e}")
                return "\n\n".join(chunk_reviews)
        
        return chunk_reviews[0]
    
    def process_moderation(self, moderator_agent: ChatOpenAI, iterations: List[Dict[str, Any]],
                          review_type: str, venue: str) -> Optional[str]:
        """Process moderator analysis of the review discussion."""
        try:
            # Get historical moderation context
            historical_context = self.context_manager.get_reviewer_context(
                "moderator", review_type, venue
            )
            
            moderator_prompt = self.create_moderator_prompt(
                iterations=iterations,
                historical_context=historical_context,
                review_type=review_type,
                venue=venue
            )
            
            response = moderator_agent.invoke([HumanMessage(content=moderator_prompt)])
            return self.extract_content(response)
            
        except Exception as e:
            logging.error(f"Error in moderation process: {e}")
            return f"Error in moderation: {str(e)}"
    
    @staticmethod
    def chunk_content(text: str, max_tokens: int = 6000) -> List[str]:
        """Split content into manageable chunks."""
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
    
    @staticmethod
    def extract_content(response: Union[str, Any]) -> str:
        """Extract content from various response types."""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'content'):
            return response.content
        elif isinstance(response, list) and len(response) > 0:
            return response[0].content
        else:
            error_msg = f"Unexpected response type: {type(response)}"
            logging.warning(error_msg)
            return f"[Error: {error_msg}]"
    
    @staticmethod
    def create_enhanced_prompt(base_prompt: str, expertise: str, review_type: str,
                             venue: str, iteration: int, previous_reviews: List[Dict[str, str]],
                             historical_context: str) -> str:
        """Create an enhanced review prompt with historical context."""
        prompt = f"""As an expert in {expertise}, you are reviewing a {review_type} for {venue}.

HISTORICAL CONTEXT:
{historical_context}

{base_prompt}"""

        if previous_reviews:
            prompt += "\n\nPrevious reviews to consider:\n"
            for prev_review in previous_reviews:
                prompt += f"\nReview by {prev_review['expertise']}:\n{prev_review['review_text']}\n"
        
        return prompt
    
    @staticmethod
    def create_moderator_prompt(iterations: List[Dict[str, Any]], historical_context: str,
                              review_type: str, venue: str) -> str:
        """Create a comprehensive moderator prompt."""
        prompt = f"""As a senior scientific moderator, analyze this {review_type} review discussion for {venue}.

HISTORICAL CONTEXT:
{historical_context}

Review Discussion Summary:
"""
        for iteration in iterations:
            prompt += f"\nIteration {iteration['iteration_number']}:\n"
            for review in iteration['reviews']:
                if review.get('success', False):
                    prompt += f"\nReview by {review['expertise']}:\n{review['review_text']}\n"
        
        prompt += """
Please provide a comprehensive analysis including:

1. DISCUSSION EVOLUTION
- How viewpoints evolved across iterations
- Key points of convergence and divergence
- Quality and depth of scientific discourse

2. REVIEW ANALYSIS
- Scientific rigor of each reviewer's contributions
- Strength of arguments and supporting evidence
- Constructiveness of the debate

3. SYNTHESIS OF KEY POINTS
- Areas of consensus
- Unresolved disagreements
- Most compelling arguments
- Critical insights gained

4. FINAL ASSESSMENT
- Overall recommendation
- Key strengths (3-5 main points)
- Key weaknesses (3-5 main points)
- Priority improvements (3-5 main suggestions)
- Final decision (Accept/Major Revision/Minor Revision/Reject)

Please provide specific examples from the discussion to support your analysis."""
        
        return prompt

def initialize_review_settings():
    """Initialize default review settings based on document type."""
    return {
        "Paper": {"reviewers": 4, "iterations": 1, "rating": "stars"},
        "Grant Proposal": {"reviewers": 3, "iterations": 2, "rating": "nih"},
        "Poster": {"reviewers": 1, "iterations": 1, "rating": "stars"}
    }

def init_app_state():
    """Initialize the application state with required managers."""
    if 'initialized' not in st.session_state:
        try:
            st.session_state.persistence_manager = ReviewPersistenceManager()
            st.session_state.context_manager = EnhancedReviewContext()
            st.session_state.review_processor = ReviewProcessor(st.session_state.context_manager)
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error initializing application state: {e}")
            raise

def show_review_history():
    """Display review history in the sidebar."""
    if 'persistence_manager' not in st.session_state:
        st.warning("Review history not available.")
        return
    
    reviews = st.session_state.persistence_manager.get_all_reviews(limit=10)
    
    if not reviews:
        st.info("No previous reviews found.")
        return
    
    for review in reviews:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(
                    f"**{review.get('document_type', 'Unknown')}** - "
                    f"{review.get('venue', 'Unknown venue')}\n\n"
                    f"*{review.get('metadata', {}).get('started_at', 'Unknown date')[:10]}*"
                )
            
            with col2:
                if st.button("View", key=f"view_{review.get('review_id', '')}"):
                    st.session_state.selected_review_id = review.get('review_id')
                    st.session_state.view_mode = 'history'

def show_selected_review():
    """Display the selected review from history."""
    if not hasattr(st.session_state, 'selected_review_id'):
        return
    
    review = st.session_state.persistence_manager.get_review_session(
        st.session_state.selected_review_id
    )
    
    if not review:
        st.warning("Selected review not found.")
        return
    
    st.markdown(f"## Viewing Review: {review.get('document_type')} - {review.get('venue')}")
    
    # Display review metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Date:** {review.get('metadata', {}).get('started_at', 'Unknown')[:10]}")
    with col2:
        st.markdown(f"**Type:** {review.get('document_type', 'Unknown')}")
    with col3:
        st.markdown(f"**Venue:** {review.get('venue', 'Unknown')}")
    
    # Display iterations
    for iteration in review.get('iterations', []):
        with st.expander(f"Iteration {iteration.get('iteration_number', '?')}", expanded=True):
            for rev in iteration.get('reviews', []):
                if rev.get('success', False):
                    st.markdown(f"### Review by {rev.get('expertise', 'Unknown Expert')}")
                    st.markdown(rev.get('review_text', 'No review text available.'))
                    st.markdown("---")
    
    # Display moderation
    if review.get('moderation'):
        with st.expander("Moderator Analysis", expanded=True):
            st.markdown(review.get('moderation'))
    
    # Download button
    if st.button("Download Review Report", key=f"download_{review.get('review_id')}"):
        report_content = generate_review_report(review)
        st.download_button(
            label="Download Report",
            data=report_content,
            file_name=f"review_report_{review.get('review_id')}.md",
            mime="text/markdown"
        )

def show_configuration_tab():
    """Display the configuration interface."""
    # Document Configuration Section
    st.markdown('<h2 class="section-header">Document Configuration</h2>', unsafe_allow_html=True)
    
    # Initialize default settings
    review_defaults = initialize_review_settings()
    
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
    
    # File Upload Section
    st.markdown("### Upload Document")
    
    uploaded_file = st.file_uploader(
        f"Upload {doc_type} (PDF format)",
        type=["pdf"],
        key="document_upload"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ {uploaded_file.name} uploaded successfully")
        with col2:
            st.info(f"Size: {uploaded_file.size / 1024:.1f} KB")
    
    # Reviewer Configuration Section
    st.markdown('<h2 class="section-header">Reviewer Configuration</h2>', unsafe_allow_html=True)
    
    # Initialize reviewer state
    if 'num_reviewers' not in st.session_state:
        st.session_state.num_reviewers = default_settings['reviewers']
    
    # Add/remove reviewer buttons
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("➕ Add", key="add_reviewer"):
            st.session_state.num_reviewers += 1
        if st.button("➖ Remove", key="remove_reviewer") and st.session_state.num_reviewers > 1:
            st.session_state.num_reviewers -= 1
    
    # Reviewer configuration
    reviewer_config = {}
    for i in range(st.session_state.num_reviewers):
        with st.expander(f"Reviewer {i+1} Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                expertise = st.text_input(
                    "Expertise",
                    value=f"Scientific Expert {i+1}",
                    key=f"expertise_{i}"
                )
            with col2:
                custom_prompt = st.text_area(
                    "Custom Instructions",
                    value=get_default_reviewer_prompt(doc_type),
                    height=150,
                    key=f"prompt_{i}"
                )
            reviewer_config[f"reviewer_{i}"] = {
                "expertise": expertise,
                "prompt": custom_prompt
            }
    
    # Number of iterations
    num_iterations = st.number_input(
        "Number of Discussion Iterations",
        min_value=1,
        max_value=5,
        value=default_settings['iterations'],
        help="Number of rounds of discussion between reviewers"
    )
    
    # Action buttons
    st.markdown("### Review Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Configuration", key="save_config", use_container_width=True):
            st.session_state.review_config = {
                "document_type": doc_type,
                "venue": venue,
                "rating_system": rating_system,
                "is_nih_grant": is_nih_grant,
                "reviewers": reviewer_config,
                "num_iterations": num_iterations,
                "bias": st.session_state.bias,
                "temperature": st.session_state.temperature
            }
            st.success("✅ Configuration saved successfully!")
    
    with col2:
        # Generate review button with dependency checks
        can_generate = uploaded_file and 'review_config' in st.session_state
        button_label = "🚀 Generate Review" if can_generate else "Upload PDF and Save Config First"
        
        if st.button(
            button_label,
            key="generate_review",
            disabled=not can_generate,
            use_container_width=True
        ):
            if not uploaded_file:
                st.error("❌ Please upload a PDF file first.")
            elif 'review_config' not in st.session_state:
                st.error("❌ Please save the configuration first.")
            else:
                with st.spinner("📊 Processing review..."):
                    try:
                        process_review(uploaded_file)
                    except Exception as e:
                        st.error(f"❌ Error during review process: {str(e)}")
                        if st.session_state.get('debug_mode', False):
                            st.exception(e)

    # Show review status
    if 'current_review' in st.session_state:
        with st.expander("Current Review Status", expanded=True):
            display_review_results(st.session_state.current_review)

def show_review_process_tab():
    """Display the review process interface."""
    st.markdown('<h2 class="section-header">Document Review</h2>', unsafe_allow_html=True)
    
    if 'review_config' not in st.session_state:
        st.warning("Please complete the configuration first.")
        return
    
    # File upload area with additional information
    st.markdown("""
    <style>
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        f"Upload {st.session_state.review_config['document_type']} (PDF)",
        type=["pdf"],
        key="uploaded_file",
        help="Upload a PDF document for review. The system will extract text and figures for analysis."
    )
    
    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "FileType": uploaded_file.type,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("File Details:", file_details)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.info("Please upload a document to begin the review process.")
        return
    
    # Configuration summary
    with st.expander("Review Configuration Summary", expanded=False):
        st.json(st.session_state.review_config)
    
    # Start review button
    if st.button("Start Review Process", disabled=not uploaded_file):
        process_review(uploaded_file)

def show_history_tab():
    """Display the review history tab."""
    st.markdown('<h2 class="section-header">Review History</h2>', unsafe_allow_html=True)
    
    # Get all reviews from context manager
    reviews = st.session_state.persistence_manager.get_all_reviews()
    
    if not reviews:
        st.info("No previous reviews found.")
        return
    
    # Filtering options
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.multiselect(
            "Filter by Document Type",
            options=list(set(review['document_type'] for review in reviews if 'document_type' in review))
        )
    
    with col2:
        filter_venue = st.multiselect(
            "Filter by Venue",
            options=list(set(review['venue'] for review in reviews if 'venue' in review))
        )
    
    # Apply filters
    filtered_reviews = [
        review for review in reviews
        if (not filter_type or review.get('document_type') in filter_type)
        and (not filter_venue or review.get('venue') in filter_venue)
    ]
    
    # Display filtered reviews
    for review in filtered_reviews:
        with st.expander(
            f"{review.get('document_type', 'Unknown')} - "
            f"{review.get('venue', 'Unknown')} - "
            f"{review.get('metadata', {}).get('started_at', 'Unknown date')[:10]}",
            expanded=False
        ):
            show_review_details(review)

def show_review_details(review: Dict[str, Any]):
    """Display detailed review information."""
    # Basic information
    st.markdown(f"**Document Type:** {review.get('document_type')}")
    st.markdown(f"**Venue:** {review.get('venue')}")
    st.markdown(f"**Review Date:** {review.get('metadata', {}).get('started_at', 'Unknown')[:10]}")
    
    # Review iterations
    for iteration in review.get('iterations', []):
        st.markdown(f"### Iteration {iteration.get('iteration_number')}")
        
        for rev in iteration.get('reviews', []):
            if rev.get('success', False):
                with st.expander(f"Review by {rev.get('expertise')}", expanded=False):
                    # Display scores if available
                    scores = extract_scores_from_review(rev.get('review_text', ''))
                    if scores:
                        st.markdown("#### Scores")
                        cols = st.columns(len(scores))
                        for col, (category, score) in zip(cols, scores.items()):
                            with col:
                                display_score(category, score)
                    
                    # Display review text
                    st.markdown(rev.get('review_text', ''))
    
    # Moderation result
    if review.get('moderation'):
        with st.expander("Moderator Analysis", expanded=False):
            st.markdown(review.get('moderation'))
    
    # Download button
    if st.button("Download Review Report", key=f"download_{review.get('review_id')}"):
        report_content = generate_review_report(review)
        st.download_button(
            label="Download Report",
            data=report_content,
            file_name=f"review_report_{review.get('review_id')}.md",
            mime="text/markdown"
        )

def extract_pdf_content(pdf_file) -> Tuple[str, List[Image.Image], Dict[str, Any]]:
    """Extract text, images, and metadata from a PDF file."""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text_content = ""
        images = []
        metadata = {
            'title': pdf_document.metadata.get('title', 'Untitled'),
            'author': pdf_document.metadata.get('author', 'Unknown'),
            'total_pages': len(pdf_document),
            'sections': []
        }
        
        # Extract section headers and content structure
        current_section = "Introduction"
        section_content = []
        
        for page_num, page in enumerate(pdf_document):
            # Extract text with formatting
            text = page.get_text("dict")
            blocks = text.get('blocks', [])
            
            for block in blocks:
                if block.get('type') == 0:  # Text block
                    font_size = block.get('size', 0)
                    font_flags = block.get('flags', 0)
                    text = block.get('text', '').strip()
                    
                    # Try to identify section headers
                    if font_size > 12 or font_flags & 16:  # Larger text or bold
                        if any(keyword in text.lower() for keyword in 
                              ['introduction', 'methods', 'results', 'discussion', 
                               'conclusion', 'abstract', 'background', 'materials']):
                            if section_content:
                                metadata['sections'].append({
                                    'title': current_section,
                                    'content_length': len('\n'.join(section_content))
                                })
                            current_section = text
                            section_content = []
                            continue
                    
                    section_content.append(text)
                    text_content += text + "\n"
            
            # Extract images
            for img in page.get_images():
                try:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
                except Exception as e:
                    logging.warning(f"Failed to extract image: {e}")
        
        # Add final section
        if section_content:
            metadata['sections'].append({
                'title': current_section,
                'content_length': len('\n'.join(section_content))
            })
        
        return text_content.strip(), images, metadata
        
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def create_review_agents(num_agents: int, review_type: str = "paper", include_moderator: bool = False) -> List[ChatOpenAI]:
    """Create review agents including a moderator if specified."""
    model = "gpt-4-turbo-preview"  # Using the latest GPT-4 model
    
    # Create regular review agents
    agents = [
        ChatOpenAI(
            temperature=st.session_state.temperature,
            openai_api_key=api_key,
            model=model
        ) for _ in range(num_agents)
    ]
    
    # Add moderator agent if requested and multiple reviewers
    if include_moderator and num_agents > 1:
        moderator_agent = ChatOpenAI(
            temperature=0.1,  # Lower temperature for moderator
            openai_api_key=api_key,
            model=model
        )
        agents.append(moderator_agent)
    
    return agents

def create_review_agents(num_agents: int, review_type: str = "paper", include_moderator: bool = False) -> List[ChatOpenAI]:
    """Create review agents including a moderator if specified."""
    model = "gpt-4-turbo-preview"  # Using the latest GPT-4 model
    
    # Create regular review agents
    agents = [
        ChatOpenAI(
            temperature=st.session_state.temperature,
            openai_api_key=api_key,
            model=model
        ) for _ in range(num_agents)
    ]
    
    # Add moderator agent if requested and multiple reviewers
    if include_moderator and num_agents > 1:
        moderator_agent = ChatOpenAI(
            temperature=0.1,  # Lower temperature for moderator
            openai_api_key=api_key,
            model=model
        )
        agents.append(moderator_agent)
    
    return agents

def display_document_analysis(metadata: Dict[str, Any], images: List[Image.Image], text_content: str):
    """Display detailed document analysis."""
    # Document statistics
    st.markdown("### Document Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", metadata['total_pages'])
    with col2:
        st.metric("Sections", len(metadata['sections']))
    with col3:
        st.metric("Figures", len(images))
    with col4:
        st.metric("Word Count", len(text_content.split()))
    
    # Section analysis
    st.markdown("### Section Details")
    for section in metadata['sections']:
        with st.expander(f"📄 {section['title']}", expanded=False):
            st.markdown(f"""
            - Content Length: {section['content_length']} characters
            - Word Count: {len(section.get('content', '').split())} words
            """)
    
    # Figure gallery
    if images:
        st.markdown("### Figure Gallery")
        gallery_cols = st.columns(3)
        for idx, img in enumerate(images):
            with gallery_cols[idx % 3]:
                st.image(img, caption=f"Figure {idx+1}", use_column_width=True)

def extract_all_scores(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all scores from review results."""
    scores = {
        'overall': {},
        'by_reviewer': {},
        'by_category': defaultdict(list)
    }
    
    for iteration in results['iterations']:
        for review in iteration['reviews']:
            if review.get('success', False):
                reviewer_scores = extract_scores_from_review(review['review_text'])
                scores['by_reviewer'][review['expertise']] = reviewer_scores
                
                for category, score in reviewer_scores.items():
                    scores['by_category'][category].append(score)
    
    # Calculate averages
    for category, category_scores in scores['by_category'].items():
        scores['overall'][category] = sum(category_scores) / len(category_scores)
    
    return scores

def display_review_results(results: Dict[str, Any]):
    """Display review results with enhanced formatting and visualization."""
    st.markdown('<h2 class="section-header">Review Results</h2>', unsafe_allow_html=True)
    
    # Overview metrics
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_score_summary(results)
        
        with col2:
            display_consensus_metrics(results)
        
        with col3:
            display_reviewer_agreement(results)
    
    # Detailed results by iteration
    for i, iteration in enumerate(results['iterations'], 1):
        with st.expander(f"Iteration {i}", expanded=True):
            display_iteration_results(iteration)
    
    # Moderator analysis
    if results.get('moderation'):
        with st.expander("Moderator Analysis", expanded=True):
            display_moderation_results(results['moderation'])

def display_score_summary(results: Dict[str, Any]):
    """Display summary of scores across all reviews."""
    st.markdown("### Score Summary")
    
    all_scores = []
    score_categories = set()
    
    # Collect all scores
    for iteration in results['iterations']:
        for review in iteration['reviews']:
            if review.get('success', False):
                scores = extract_scores_from_review(review['review_text'])
                if scores:
                    all_scores.append(scores)
                    score_categories.update(scores.keys())
    
    if all_scores:
        # Calculate averages
        averages = {}
        for category in score_categories:
            scores = [s[category] for s in all_scores if category in s]
            if scores:
                averages[category] = sum(scores) / len(scores)
        
        # Display scores
        for category, avg in averages.items():
            st.metric(
                label=category.title(),
                value=f"{avg:.1f}",
                delta=None
            )
    else:
        st.info("No numerical scores found in reviews.")

def display_consensus_metrics(results: Dict[str, Any]):
    """Display metrics about reviewer consensus and agreement."""
    st.markdown("### Consensus Analysis")
    
    # Analyze key points across reviews
    key_points = defaultdict(int)
    for iteration in results['iterations']:
        for review in iteration['reviews']:
            if review.get('success', False):
                points = extract_key_points(review['review_text'])
                for point in points:
                    key_points[point.lower()] += 1
    
    # Display top consensus points
    if key_points:
        st.markdown("#### Top Consensus Points")
        sorted_points = sorted(key_points.items(), key=lambda x: x[1], reverse=True)[:5]
        for point, count in sorted_points:
            st.markdown(f"- {point} ({count} mentions)")
    else:
        st.info("No consensus points identified.")

def display_reviewer_agreement(results: Dict[str, Any]):
    """Display analysis of reviewer agreement and disagreement."""
    st.markdown("### Reviewer Agreement")
    
    # Calculate agreement metrics
    agreements = analyze_reviewer_agreement(results)
    
    if agreements:
        agreement_level = calculate_agreement_level(agreements)
        st.metric(
            label="Overall Agreement Level",
            value=f"{agreement_level:.1f}%",
            delta=None
        )
        
        # Display key agreements and disagreements
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Key Agreements")
            for point in agreements.get('agreements', [])[:3]:
                st.markdown(f"- {point}")
        
        with col2:
            st.markdown("#### Key Disagreements")
            for point in agreements.get('disagreements', [])[:3]:
                st.markdown(f"- {point}")
    else:
        st.info("Insufficient data to analyze reviewer agreement.")

def display_iteration_results(iteration: Dict[str, Any]):
    """Display results from a single iteration."""
    for review in iteration['reviews']:
        if review.get('success', False):
            with st.container():
                st.markdown(f"### Review by {review['expertise']}")
                
                # Extract and display scores
                scores = extract_scores_from_review(review['review_text'])
                if scores:
                    st.markdown("#### Scores")
                    cols = st.columns(len(scores))
                    for col, (category, score) in zip(cols, scores.items()):
                        with col:
                            display_score(category, score)
                
                # Display main review content
                sections = split_review_sections(review['review_text'])
                for section_title, content in sections.items():
                    with st.expander(section_title, expanded=True):
                        st.markdown(content)
                
                # Display metadata
                st.markdown(f"*Reviewed at: {review['timestamp']}*")
        else:
            st.error(f"Error in review by {review['expertise']}: {review.get('error', 'Unknown error')}")

def display_moderation_results(moderation: str):
    """Display moderator analysis with enhanced formatting."""
    sections = split_moderation_sections(moderation)
    
    # Display overall recommendation first
    if 'Final Assessment' in sections:
        st.markdown("### Final Assessment")
        st.markdown(sections['Final Assessment'])
    
    # Display other sections
    for title, content in sections.items():
        if title != 'Final Assessment':
            with st.expander(title, expanded=True):
                st.markdown(content)

def analyze_reviewer_agreement(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze agreement and disagreement between reviewers."""
    all_points = []
    agreements = []
    disagreements = []
    
    # Collect all key points and scores
    for iteration in results['iterations']:
        iteration_points = []
        for review in iteration['reviews']:
            if review.get('success', False):
                points = extract_key_points(review['review_text'])
                iteration_points.append(points)
        
        # Find agreements and disagreements within iteration
        if iteration_points:
            common_points = set.intersection(*map(set, iteration_points))
            unique_points = set.union(*map(set, iteration_points)) - common_points
            
            agreements.extend(common_points)
            disagreements.extend(unique_points)
    
    return {
        'agreements': list(set(agreements)),
        'disagreements': list(set(disagreements))
    }

def calculate_agreement_level(agreements: Dict[str, List[str]]) -> float:
    """Calculate overall agreement level as a percentage."""
    total_points = len(agreements['agreements']) + len(agreements['disagreements'])
    if total_points == 0:
        return 0.0
    return (len(agreements['agreements']) / total_points) * 100

def split_review_sections(review_text: str) -> Dict[str, str]:
    """Split review text into logical sections."""
    sections = {}
    current_section = "Overview"
    current_content = []
    
    for line in review_text.split('\n'):
        if line.strip().upper() == line.strip() and line.strip() and ':' in line:
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = line.strip().rstrip(':')
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

def split_moderation_sections(moderation_text: str) -> Dict[str, str]:
    """Split moderator analysis into sections."""
    sections = {}
    current_section = "Overview"
    current_content = []
    
    for line in moderation_text.split('\n'):
        if line.strip().upper() == line.strip() and line.strip() and ':' in line:
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = line.strip().rstrip(':')
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

def display_score(category: str, score: Union[float, int, str]):
    """Display a score with appropriate formatting."""
    if isinstance(score, str) and '★' in score:
        st.markdown(f"**{category}**")
        st.markdown(f"<h3 class='score-display'>{score}</h3>", unsafe_allow_html=True)
    else:
        st.metric(
            label=category.title(),
            value=f"{float(score):.1f}",
            delta=None
        )

def generate_review_report(review_data: Dict[str, Any]) -> str:
    """Generate a formatted markdown report of the review results."""
    report = f"""# Scientific Review Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Document Information
- Type: {review_data['document_type']}
- Venue: {review_data['venue']}
- Number of Reviewers: {len(review_data.get('expertises', []))}
- Number of Iterations: {review_data.get('num_iterations', 0)}

## Review Results
"""
    
    # Add iteration results
    for i, iteration in enumerate(review_data.get('iterations', []), 1):
        report += f"\n### Iteration {i}\n"
        for review in iteration['reviews']:
            if review.get('success', False):
                report += f"\n#### Review by {review['expertise']}\n"
                report += review['review_text']
                report += "\n---\n"
    
    # Add moderator analysis
    if review_data.get('moderation'):
        report += "\n## Moderator Analysis\n"
        report += review_data['moderation']
    
    # Add analysis summary
    if 'analysis' in review_data:
        report += "\n## Analysis Summary\n"
        report += generate_analysis_summary(review_data['analysis'])
    
    return report

def generate_analysis_summary(analysis: Dict[str, Any]) -> str:
    """Generate a summary of the review analysis."""
    summary = """### Key Metrics\n"""
    
    if 'agreement_level' in analysis:
        summary += f"- Reviewer Agreement Level: {analysis['agreement_level']:.1f}%\n"
    
    if 'consensus_points' in analysis:
        summary += "\n### Consensus Points\n"
        for point in analysis['consensus_points']:
            summary += f"- {point}\n"
    
    if 'key_recommendations' in analysis:
        summary += "\n### Key Recommendations\n"
        for rec in analysis['key_recommendations']:
            summary += f"- {rec}\n"
    
    return summary

def get_default_reviewer_prompt(doc_type: str) -> str:
    """Get default prompt for a reviewer based on document type."""
    prompts = get_default_prompts()
    return prompts.get(doc_type, {}).get('reviewer', "Please provide a thorough review.")

def extract_key_points(text: str) -> List[str]:
    """Extract key points from review text."""
    points = []
    patterns = [
        r'(?:•|\*|\-|\d+\.)\s*([^\n]+)',
        r'key\s+points?:([^\n]+)',
        r'strengths?:([^\n]+)',
        r'weaknesses?:([^\n]+)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            point = match.group(1).strip()
            if point:
                points.append(point)
    
    return points

def extract_scores_from_review(review_text: str) -> Dict[str, Union[float, str]]:
    """Extract scores from review text with support for different formats."""
    scores = {}
    patterns = [
        r'(\w+)\s*(?:score|rating):\s*(\d+(?:\.\d+)?)/?\d*',
        r'(\w+):\s*(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+',
        r'(\w+):\s*([★]+(?:☆)*)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, review_text, re.IGNORECASE)
        for match in matches:
            category = match.group(1).strip()
            score = match.group(2)
            if '★' in str(score):
                scores[category] = score
            else:
                try:
                    scores[category] = float(score)
                except ValueError:
                    continue
    
    return scores

def initialize_session_state():
    """Initialize session state variables with default values."""
    if 'bias' not in st.session_state:
        st.session_state.bias = 0
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.1
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'num_reviewers' not in st.session_state:
        st.session_state.num_reviewers = 4

def main_content():
    """Main application content."""
    # Initialize session state
    initialize_session_state()
    
    # Initialize application state
    init_app_state()
    
    st.markdown('<h1 class="main-header">Multi-Agent Scientific Review System</h1>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("Review Configuration")
        
        # Reviewer bias slider
        st.session_state.bias = st.slider(
            "Reviewer Bias",
            min_value=-2,
            max_value=2,
            value=st.session_state.bias,
            help="""
            -2: Extremely negative and biased
            -1: Somewhat negative and biased
            0: Unbiased/objective
            1: Positive and enthusiastic
            2: Extremely positive and passionate
            """
        )
        
        # View previous reviews
        with st.expander("Review History", expanded=False):
            show_review_history()
        
        # Advanced settings
        with st.expander("Advanced Settings", expanded=False):
            st.session_state.temperature = st.slider(
                "Model Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Controls randomness in model responses"
            )
            
            st.session_state.debug_mode = st.checkbox(
                "Debug Mode",
                value=st.session_state.debug_mode,
                help="Show detailed logging information"
            )
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Configure & Review", "Review History"])
    
    with tab1:
        show_configuration_tab()
    
    with tab2:
        show_history_tab()

if __name__ == "__main__":
    try:
        main_content()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Unexpected error in main application:")
