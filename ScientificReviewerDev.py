import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import UnstructuredPowerPointLoader
import fitz
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import re

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
    },
    "Poster": {
        "reviewers": 2,
        "iterations": 1,
        "scoring": "stars",
        "scale": "1-5 stars"
    },
    "Presentation": {
        "reviewers": 2,
        "iterations": 1,
        "scoring": "stars",
        "scale": "1-5 stars"
    }
}

def extract_document_content(uploaded_file, file_type: str) -> tuple[str, List[Dict[str, Any]]]:
    """Extract content from various document types with structure preservation."""
    try:
        if file_type == "pdf":
            return extract_pdf_content(uploaded_file)
        elif file_type == "pptx":
            return extract_pptx_content(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        raise Exception(f"Error processing {file_type.upper()}: {str(e)}")

def extract_pdf_content(pdf_file) -> tuple[str, List[Dict[str, Any]]]:
    """Extract text content from PDF file with structure."""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text_content = ""
        sections = []
        
        for page_num, page in enumerate(pdf_document):
            text = page.get_text()
            # Try to identify section headers
            lines = text.split('\n')
            current_section = None
            section_content = []
            
            for line in lines:
                # Simple heuristic for section headers - can be improved
                if line.isupper() and len(line.strip()) > 3:
                    if current_section and section_content:
                        sections.append({
                            'type': 'section',
                            'title': current_section,
                            'content': '\n'.join(section_content)
                        })
                        section_content = []
                    current_section = line.strip()
                else:
                    section_content.append(line)
            
            # Add remaining content
            if current_section and section_content:
                sections.append({
                    'type': 'section',
                    'title': current_section,
                    'content': '\n'.join(section_content)
                })
            
            # Add page as a section if no sections were identified
            if not sections:
                sections.append({
                    'type': 'page',
                    'number': page_num + 1,
                    'content': text
                })
            
            text_content += text
            
        return text_content.strip(), sections
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def extract_pptx_content(pptx_file) -> tuple[str, List[Dict[str, Any]]]:
    """Extract content from PowerPoint with slide preservation."""
    try:
        # Save uploaded file temporarily
        with open("temp.pptx", "wb") as f:
            f.write(pptx_file.getvalue())
        
        # Load and process slides
        loader = UnstructuredPowerPointLoader("temp.pptx")
        documents = loader.load()
        
        full_content = ""
        sections = []
        
        for idx, doc in enumerate(documents):
            content = doc.page_content
            full_content += f"\nSlide {idx + 1}:\n{content}\n"
            sections.append({
                'type': 'slide',
                'number': idx + 1,
                'content': content,
                'metadata': doc.metadata
            })
        
        import os
        os.remove("temp.pptx")  # Clean up
        
        return full_content.strip(), sections
    except Exception as e:
        raise Exception(f"Error processing PowerPoint: {str(e)}")

def parse_review_sections(content: str) -> Dict[str, str]:
    """Parse review content into structured sections with cleaned formatting."""
    sections = {}
    
    # Define section markers and their keys
    section_markers = {
        'RESPONSE TO PREVIOUS REVIEWS': 'response',
        'SECTION-BY-SECTION ANALYSIS': 'analysis',
        'SIGNIFICANCE EVALUATION': 'significance',
        'INNOVATION ASSESSMENT': 'innovation',
        'APPROACH ANALYSIS': 'approach',
        'SCORING': 'scoring',
        'RECOMMENDATIONS': 'recommendations'
    }
    
    # Find each section's content
    for marker, key in section_markers.items():
        if marker in content:
            start = content.find(marker)
            # Find the start of the next section or end of content
            next_section_start = float('inf')
            for other_marker in section_markers:
                if other_marker != marker:
                    pos = content.find(other_marker, start + len(marker))
                    if pos != -1 and pos < next_section_start:
                        next_section_start = pos
            
            if next_section_start == float('inf'):
                next_section_start = len(content)
            
            section_content = content[start + len(marker):next_section_start].strip(':').strip()
            
            # Clean up the content
            section_content = re.sub(r'\*\*(.*?)\*\*', r'\1', section_content)  # Remove bold markers
            section_content = section_content.replace('*.', '•')  # Replace asterisk bullets
            section_content = re.sub(r'\n{3,}', '\n\n', section_content)  # Remove extra newlines
            
            if section_content:
                sections[key] = section_content
    
    return sections

def parse_moderator_sections(content: str) -> Dict[str, str]:
    """Parse moderator summary into structured sections."""
    sections = {}
    
    # Define section markers and their keys
    section_markers = {
        'KEY POINTS OF AGREEMENT': 'agreement',
        'POINTS OF CONTENTION': 'contention',
        'DISCUSSION EVOLUTION': 'evolution',
        'FINAL SYNTHESIS': 'synthesis'
    }
    
    # Find each section's content
    for marker, key in section_markers.items():
        if marker in content:
            start = content.find(marker)
            # Find the start of the next section or end of content
            next_section_start = float('inf')
            for other_marker in section_markers:
                if other_marker != marker:
                    pos = content.find(other_marker, start + len(marker))
                    if pos != -1 and pos < next_section_start:
                        next_section_start = pos
            
            if next_section_start == float('inf'):
                next_section_start = len(content)
            
            section_content = content[start + len(marker):next_section_start].strip(':').strip()
            if section_content:
                sections[key] = section_content
    
    return sections

def parse_slide_review(content: str) -> Dict[str, str]:
    """Parse slide review content into sections."""
    sections = {}
    section_markers = {
        'SLIDE PURPOSE': '🎯 Slide Purpose',
        'CONTENT ANALYSIS': '📝 Content Analysis',
        'VISUAL DESIGN': '🎨 Visual Design',
        'COMMUNICATION': '💬 Communication',
        'RECOMMENDATIONS': '📋 Recommendations'
    }
    
    for marker, title in section_markers.items():
        if marker in content:
            start = content.find(marker)
            end = len(content)
            for next_marker in section_markers:
                next_pos = content.find(next_marker, start + len(marker))
                if next_pos != -1 and next_pos < end:
                    end = next_pos
            section_content = content[start + len(marker):end].strip(':').strip()
            if section_content:
                sections[title] = section_content
    
    return sections

def display_review_sections(sections: Dict[str, str]):
    """Display review sections with improved formatting."""
    section_order = [
        'response', 'analysis', 'significance', 'innovation', 
        'approach', 'scoring', 'recommendations'
    ]
    
    icons = {
        'response': '💬',
        'analysis': '📝',
        'significance': '🎯',
        'innovation': '💡',
        'approach': '🔍',
        'scoring': '⭐',
        'recommendations': '📋'
    }
    
    titles = {
        'response': 'Response to Previous Reviews',
        'analysis': 'Section Analysis',
        'significance': 'Significance Evaluation',
        'innovation': 'Innovation Assessment',
        'approach': 'Approach Analysis',
        'scoring': 'Scoring',
        'recommendations': 'Recommendations'
    }
    
    for section in section_order:
        if section in sections:
            content = sections[section]
            
            # Clean up the content
            content = content.replace('**', '')  # Remove unnecessary asterisks
            content = re.sub(r'\n{3,}', '\n\n', content)  # Remove extra newlines
            content = content.strip()
            
            st.markdown(f"### {icons[section]} {titles[section]}")
            
            if section == 'analysis':
                # Special handling for section analysis to format subsections
                for line in content.split('\n'):
                    if ':' in line:
                        part, desc = line.split(':', 1)
                        if 'Content Summary' in part:
                            st.markdown(f"**{part.strip()}:**{desc}")
                            st.markdown("---")
                        elif 'Critical Changes' in part:
                            st.markdown(f"**{part.strip()}:**{desc}")
                            st.markdown("---")
                        elif 'Suggested Improvements' in part:
                            st.markdown(f"**{part.strip()}:**{desc}")
                            st.markdown("---")
                        elif 'Specific Line Edits' in part:
                            st.markdown(f"**{part.strip()}:**{desc}")
                            st.markdown("---")
                        else:
                            st.markdown(line)
                    else:
                        if line.strip():
                            st.markdown(line)
            
            elif section == 'scoring':
                # Special handling for scoring section
                for line in content.split('\n'):
                    if '★' in line:
                        category, score = line.split(':', 1)
                        st.markdown(f"**{category.strip()}:** {score.strip()}")
                    else:
                        if line.strip():
                            st.markdown(line)
            
            elif section == 'recommendations':
                # Special handling for recommendations
                current_section = None
                for line in content.split('\n'):
                    if 'Critical changes needed:' in line:
                        current_section = "Required Changes"
                        st.markdown(f"**{current_section}:**")
                    elif 'Suggested improvements:' in line:
                        current_section = "Optional Improvements"
                        st.markdown(f"**{current_section}:**")
                    else:
                        if line.strip() and not line.startswith('**'):
                            st.markdown(f"- {line.strip()}")
            
            else:
                st.markdown(content)
            
            st.markdown("---")

def display_moderator_sections(content: str):
    """Display moderator summary sections with consistent formatting."""
    sections = parse_moderator_sections(content)
    
    section_order = [
        'agreement', 
        'contention', 
        'evolution', 
        'synthesis'
    ]
    
    icons = {
        'agreement': '🤝',
        'contention': '⚖️',
        'evolution': '📈',
        'synthesis': '🎯'
    }
    
    titles = {
        'agreement': 'Points of Agreement',
        'contention': 'Points of Contention',
        'evolution': 'Discussion Evolution',
        'synthesis': 'Final Synthesis'
    }
    
    for section in section_order:
        if section in sections:
            st.markdown(f"### {icons[section]} {titles[section]}")
            st.markdown(sections[section])
            st.markdown("---")

    if not any(section in sections for section in section_order):
        st.markdown(content)  # Fallback: display raw content if parsing fails

def display_review_results(results: Dict[str, Any]):
    """Display review results with improved formatting and error handling."""
    st.markdown("## Review Results")
    
    if not results.get('iterations'):
        st.warning("No review results available.")
        return
    
    # Create tabs only if there are actual iterations
    valid_iterations = [i for i in results['iterations'] if i['reviews']]
    if not valid_iterations:
        st.warning("No valid reviews to display.")
        return
        
    tabs = st.tabs([f"Iteration {i+1}" for i in range(len(valid_iterations))])
    
    for idx, (tab, iteration) in enumerate(zip(tabs, valid_iterations)):
        with tab:
            for review in iteration['reviews']:
                with st.expander(f"Review by {review['reviewer']}", expanded=True):
                    if review.get('error', False):
                        st.error(review['content'])
                    elif review.get('is_presentation', False):
                        # Display presentation review
                        st.markdown("### 📊 Overall Presentation Analysis")
                        st.markdown(review['content'])
                        st.markdown("---")
                        
                        # Only create slide tabs if there are valid slide reviews
                        valid_slides = [sr for sr in review['slide_reviews'] if not sr.get('error')]
                        if valid_slides:
                            st.markdown("### 🎯 Slide-by-Slide Review")
                            slide_tabs = st.tabs([f"Slide {sr['slide_number']}" for sr in valid_slides])
                            
                            for slide_tab, slide_review in zip(slide_tabs, valid_slides):
                                with slide_tab:
                                    sections = parse_slide_review(slide_review['content'])
                                    
                                    for section_title, content in sections.items():
                                        st.markdown(f"#### {section_title}")
                                        st.markdown(content)
                                        st.markdown("---")
                                    
                                    if slide_review.get('scores'):
                                        st.markdown("#### Scores")
                                        for category, score in slide_review['scores'].items():
                                            st.markdown(f"**{category}:** {'★' * int(score)}{'☆' * (5 - int(score))}")
                        
                        if review.get('structure_analysis'):
                            st.markdown("### 📈 Presentation Structure")
                            for item in review['structure_analysis']:
                                st.markdown(f"**Slide {item['slide_number']} ({item['content_type']}):**")
                                if item.get('key_points'):
                                    for point in item['key_points']:
                                        st.markdown(f"- {point}")
                    else:
                        # Display regular review content
                        sections = parse_review_sections(review['content'])
                        display_review_sections(sections)
                    
                    st.markdown(f"*Reviewed at: {review['timestamp']}*")
    
    # Display moderator summary
    if results.get('moderator_summary'):
        st.markdown("## 🎯 Moderator Analysis")
        display_moderator_sections(results['moderator_summary'])

class ModeratorAgent:
    def __init__(self, model="gpt-4o"):  # Update default model
        self.model = ChatOpenAI(
            temperature=0.0,
            openai_api_key=st.secrets["openai_api_key"],
            model=model
        )
    
    def summarize_discussion(self, iterations: List[Dict[str, Any]]) -> str:
        """Analyze the evolution of discussion across iterations."""
        discussion_summary = "Discussion Evolution Summary:\n\n"
        for iteration in iterations:
            discussion_summary += f"Iteration {iteration['iteration_number']} reviews:\n"
            for review in iteration['reviews']:
                if not review.get('error', False):  # Only include successful reviews
                    discussion_summary += f"- {review['reviewer']}: {review['content']}\n\n"
        
        moderator_prompt = f"""As a moderator, analyze how the scientific discussion evolved across these iterations:

{discussion_summary}

Please provide a comprehensive analysis using the following structure:

1. KEY POINTS OF AGREEMENT:
   - Areas where reviewers reached consensus
   - Shared concerns or praise
   - Common recommendations

2. POINTS OF CONTENTION:
   - Areas of disagreement
   - Differing perspectives
   - Varying interpretations

3. DISCUSSION EVOLUTION:
   - How viewpoints changed across iterations
   - How reviewers responded to each other
   - Development of key arguments

4. FINAL SYNTHESIS:
   - Overall consensus
   - Key recommendations
   - Final decision recommendation
   - Critical next steps

Format your response using these exact sections and maintain a balanced, objective perspective."""

        try:
            response = self.model.invoke([HumanMessage(content=moderator_prompt)])
            return response.content
        except Exception as e:
            logging.error(f"Error in moderation: {e}")
            return f"Error generating moderation summary: {str(e)}"

class ReviewManager:
    def __init__(self):
        self.model_config = {
            "Paper": "gpt-4o",
            "Grant": "gpt-4o",
            "Poster": "gpt-4o",
            "Presentation": "gpt-4o"
        }
        self.moderator = ModeratorAgent()
    
    def _create_previous_context(self, reviews: List[Dict[str, Any]]) -> str:
        """Create context from previous reviews."""
        if not reviews:
            return ""
            
        context = "\nPrevious reviews:\n"
        for review in reviews:
            if not review.get('error', False):
                if review.get('is_presentation', False):
                    # Handle presentation reviews differently
                    context += f"\n{review['reviewer']} Overall Assessment:\n{review['content']}\n"
                    
                    # Add key points from slide reviews
                    context += "\nKey points from slides:\n"
                    for slide_review in review.get('slide_reviews', []):
                        if not slide_review.get('error'):
                            context += f"Slide {slide_review['slide_number']}: "
                            context += f"{'; '.join(slide_review.get('key_points', []))}\n"
                else:
                    # Handle regular document reviews
                    context += f"\n{review['reviewer']}:\n{review['content']}\n"
        
        return context

    def process_review(self, content: str, sections: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process document review with multiple iterations and moderation."""
        iterations = []
        all_reviews = []
        
        try:
            is_presentation = any(section.get('type') == 'slide' for section in sections)
            if is_presentation:
                config['doc_type'] = "Presentation"
            
            for iteration in range(config['iterations']):
                iteration_reviews = []
                previous_context = self._create_previous_context(all_reviews)
                
                for reviewer in config['reviewers']:
                    try:
                        temperature = self._calculate_temperature(config.get('bias', 0))
                        model = self.model_config.get(config['doc_type'], "gpt-4o")
                        
                        if is_presentation:
                            presentation_reviewer = PresentationReviewer(model=model)
                            review_results = presentation_reviewer.review_presentation(
                                sections=sections,
                                expertise=reviewer['expertise']
                            )
                            
                            if not review_results.get('overall_review', {}).get('content'):
                                raise ValueError("Missing review content from presentation reviewer")
                            
                            review = {
                                'reviewer': reviewer['expertise'],
                                'content': review_results['overall_review']['content'],
                                'slide_reviews': review_results['slide_reviews'],
                                'structure_analysis': review_results['structure_analysis'],
                                'timestamp': datetime.now().isoformat(),
                                'is_presentation': True
                            }
                        else:
                            agent = ChatOpenAI(
                                temperature=temperature,
                                openai_api_key=st.secrets["openai_api_key"],
                                model=model
                            )
                            review = self._process_full_content(
                                agent=agent,
                                content=content,
                                reviewer=reviewer,
                                config=config,
                                iteration=iteration + 1,
                                previous_context=previous_context
                            )
                        
                        iteration_reviews.append(review)
                        all_reviews.append(review)
                        
                    except Exception as e:
                        error_msg = f"Error in review by {reviewer['expertise']}: {str(e)}"
                        logging.error(error_msg)
                        iteration_reviews.append({
                            'reviewer': reviewer['expertise'],
                            'content': f"Review failed: {str(e)}",
                            'timestamp': datetime.now().isoformat(),
                            'error': True
                        })
                
                iterations.append({
                    'iteration_number': iteration + 1,
                    'reviews': iteration_reviews
                })
            
            # Generate moderator summary if we have valid reviews
            if any(not review.get('error', False) for reviews in iterations for review in reviews['reviews']):
                moderator_summary = self.moderator.summarize_discussion(iterations)
            else:
                moderator_summary = "No valid reviews to summarize"
            
            return {
                'iterations': iterations,
                'moderator_summary': moderator_summary,
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'doc_type': config['doc_type'],
                'is_presentation': is_presentation
            }
            
        except Exception as e:
            error_msg = f"Error in review process: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
            
    def _calculate_temperature(self, bias: int) -> float:
        """Calculate temperature based on reviewer bias while keeping it within valid range."""
        # Base temperature of 0.7, adjusted by bias but kept within [0.0, 1.0]
        return max(0.0, min(1.0, 0.7 + (bias * 0.1)))

class ReviewManager(ReviewManager):  # Continuing the ReviewManager class
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

    def _create_section_review_prompt(self, doc_type: str, expertise: str, scoring_type: str,
                                    iteration: int, previous_context: str, 
                                    section_type: str, section_number: int) -> str:
        """Create review prompt for individual sections."""
        base_prompt = f"""As a {expertise}, please review this {section_type} {section_number} of the {doc_type}.
{previous_context}

Please analyze this {section_type} using the following structure:"""

        if doc_type.lower() == "poster":
            return base_prompt + """
1. CONTENT ANALYSIS:
   - Key messages and findings
   - Scientific accuracy
   - Data presentation
   - Visual effectiveness

2. SPECIFIC RECOMMENDATIONS:
   [REQUIRED] Critical improvements needed
   [OPTIONAL] Suggested enhancements

3. SECTION SCORE (★☆):
   Rate this section (1-5 stars):
   - Content Quality
   - Visual Design
   - Communication Effectiveness"""
        
        else:  # PowerPoint slides
            return base_prompt + """
1. SLIDE ANALYSIS:
   - Main points
   - Clarity and organization
   - Visual elements
   - Text content

2. RECOMMENDATIONS:
   [REQUIRED] Critical improvements
   [OPTIONAL] Suggested enhancements

3. SLIDE EFFECTIVENESS (★☆):
   Rate these aspects (1-5 stars):
   - Content Clarity
   - Visual Design
   - Presentation Impact"""

    def _create_compilation_prompt(self, doc_type: str, expertise: str, 
                                 section_reviews: List[Dict[str, Any]]) -> str:
        """Create prompt for compiling section reviews."""
        sections_summary = "\n\n".join([
            f"Section {review['section_number']}:\n{review['content']}"
            for review in section_reviews
        ])
        
        return f"""As a {expertise}, please compile your individual section reviews into a coherent overall review:

Previous section reviews:
{sections_summary}

Please provide a comprehensive review using this structure:

1. OVERALL ANALYSIS:
   - Key strengths across all sections
   - Major areas for improvement
   - Coherence and flow

2. SECTION-BY-SECTION SUMMARY:
   - Brief summary of key points for each section
   - Critical changes needed
   - Suggested improvements

3. OVERALL SCORING (★☆):
   Rate each category (1-5 stars):
   - Content Quality
   - Visual Design
   - Communication Effectiveness
   - Overall Impact

4. FINAL RECOMMENDATIONS:
   [REQUIRED] Critical changes needed
   [OPTIONAL] Suggested improvements"""

class PresentationReviewer:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = ChatOpenAI(
            temperature=0.1,
            openai_api_key=st.secrets["openai_api_key"],
            model=model
        )

    def review_presentation(self, sections: List[Dict[str, Any]], expertise: str) -> Dict[str, Any]:
        """Review a presentation slide by slide with comprehensive analysis."""
        try:
            if not sections:
                raise ValueError("No sections provided for review")

            slide_reviews = []
            overall_structure = []

            # Review each slide
            for section in sections:
                if section.get('type') == 'slide':
                    if not section.get('content'):
                        logging.warning(f"Skipping slide {section.get('number', '?')} - no content")
                        continue

                    review = self._review_slide(section, expertise)
                    if not review.get('error'):
                        formatted_content = self._format_review_text(review['content'])
                        review['content'] = formatted_content
                        slide_reviews.append(review)
                        
                        overall_structure.append({
                            'slide_number': section.get('number', len(overall_structure) + 1),
                            'content_type': self._identify_slide_type(section['content']),
                            'key_points': review.get('key_points', [])
                        })
                    else:
                        logging.error(f"Error in slide {section.get('number', '?')}: {review.get('error')}")

            if not slide_reviews:
                raise ValueError("No valid slides were processed")

            # Generate overall presentation review
            overall_review = self._generate_overall_review(slide_reviews, overall_structure, expertise)
            if not overall_review or not overall_review.get('content'):
                raise ValueError("Failed to generate overall review")

            overall_review['content'] = self._format_review_text(overall_review['content'])

            return {
                'slide_reviews': slide_reviews,
                'overall_review': overall_review,
                'structure_analysis': overall_structure
            }

        except Exception as e:
            error_msg = f"Presentation review failed: {str(e)}"
            logging.error(error_msg)
            return {
                'error': True,
                'message': error_msg,
                'slide_reviews': [],
                'overall_review': {'content': error_msg},
                'structure_analysis': []
            }

    def _identify_slide_type(self, content: str) -> str:
        """Identify the type/purpose of a slide."""
        try:
            if not content:
                return 'unknown'

            content_lower = content.lower()
            slide_types = {
                'title': ['title', 'agenda', 'outline', 'overview'],
                'introduction': ['introduction', 'background', 'context'],
                'methods': ['methods', 'methodology', 'approach'],
                'results': ['results', 'findings', 'data'],
                'discussion': ['discussion', 'implications'],
                'conclusion': ['conclusion', 'summary', 'takeaways'],
                'references': ['references', 'citations']
            }

            for slide_type, keywords in slide_types.items():
                if any(keyword in content_lower for keyword in keywords):
                    return slide_type
            return 'content'  # default type

        except Exception as e:
            logging.error(f"Error identifying slide type: {str(e)}")
            return 'unknown'

    def _format_review_text(self, raw_content: str) -> str:
        """Format and clean review text for better presentation."""
        try:
            # Remove excessive newlines
            content = re.sub(r'\n{3,}', '\n\n', raw_content)

            # Ensure consistent heading formatting
            content = re.sub(r'(?m)^([A-Z][A-Z\s]+):$', r'\n\1:', content)

            # Clean up bullet points
            content = re.sub(r'(?m)^\s*[*•]\s*', '• ', content)

            # Ensure consistent star formatting
            content = content.replace('*', '★')
            content = re.sub(r'(?<=\d)\s*stars?\b', '★', content)

            return content.strip()
        except Exception as e:
            logging.error(f"Error formatting review text: {str(e)}")
            return raw_content

    def _review_slide(self, slide: Dict[str, Any], expertise: str) -> Dict[str, Any]:
        """Review an individual slide with comprehensive analysis."""
        try:
            if not slide.get('content'):
                raise ValueError("Empty slide content")

            prompt = f"""As a {expertise}, review this presentation slide:

Slide {slide.get('number', 1)} Content:
{slide['content']}

Provide a structured analysis using these exact sections:

SLIDE PURPOSE:
- Main message and objective
- Target audience relevance
- Position in presentation flow

CONTENT ANALYSIS:
- Key points and main ideas
- Technical accuracy
- Supporting evidence
- Data presentation (if applicable)

VISUAL DESIGN:
- Layout effectiveness
- Visual hierarchy
- Space utilization
- Graphics and images effectiveness

COMMUNICATION:
- Clarity of message
- Text conciseness
- Audience engagement
- Technical language appropriateness

RECOMMENDATIONS:
[REQUIRED] Critical improvements needed:
- List specific required changes

[OPTIONAL] Enhancement suggestions:
- List potential improvements

SCORING:
Rate each aspect (using ★):
Content Quality: (1-5 stars)
Visual Design: (1-5 stars)
Communication Effectiveness: (1-5 stars)"""

            response = self.client.invoke([HumanMessage(content=prompt)])

            if not response or not hasattr(response, 'content'):
                raise ValueError("Invalid AI response")

            content = response.content

            # Extract structured information
            review = {
                'slide_number': slide.get('number', 1),
                'content': content,
                'scores': self._extract_scores(content),
                'key_points': self._extract_key_points(content),
                'recommendations': self._extract_recommendations(content)
            }

            return review

        except Exception as e:
            error_msg = f"Slide review failed: {str(e)}"
            logging.error(error_msg)
            return {
                'slide_number': slide.get('number', 1),
                'error': error_msg,
                'content': f"Review failed for slide {slide.get('number', 1)}: {str(e)}"
            }

    def _generate_overall_review(self, slide_reviews: List[Dict[str, Any]], 
                               structure: List[Dict[str, Any]], expertise: str) -> Dict[str, Any]:
        """Generate overall presentation review."""
        try:
            if not slide_reviews:
                raise ValueError("No slide reviews to analyze")

            # Create a summary of the presentation structure
            structure_summary = "\n".join([
                f"Slide {s['slide_number']}: {s['content_type'].title()}"
                for s in structure
            ])

            prompt = f"""As a {expertise}, synthesize this complete presentation review:

Presentation Structure:
{structure_summary}

Individual Slide Reviews Summary:
{json.dumps([{
    'slide_number': r['slide_number'],
    'key_points': r.get('key_points', []),
    'scores': r.get('scores', {})
} for r in slide_reviews if not r.get('error')], indent=2)}

Provide a comprehensive review using these exact sections:

OVERALL ASSESSMENT:
- Main objectives and target audience
- Presentation flow and structure
- Technical depth and clarity

KEY STRENGTHS:
- List major effective elements
- Highlight successful aspects

CRITICAL IMPROVEMENTS:
- List essential changes needed
- Identify major issues

RECOMMENDATIONS:
[REQUIRED] Essential improvements:
- List critical changes needed

[OPTIONAL] Enhancements:
- List suggested improvements

OVERALL SCORES (using ★):
Content Organization: (1-5 stars)
Visual Design: (1-5 stars)
Technical Quality: (1-5 stars)
Presentation Impact: (1-5 stars)"""

            response = self.client.invoke([HumanMessage(content=prompt)])

            if not response or not hasattr(response, 'content'):
                raise ValueError("Invalid AI response for overall review")

            return {
                'content': response.content,
                'scores': self._extract_scores(response.content),
                'recommendations': self._extract_recommendations(response.content)
            }

        except Exception as e:
            error_msg = f"Overall review generation failed: {str(e)}"
            logging.error(error_msg)
            return {
                'content': error_msg,
                'scores': {},
                'recommendations': {'required': [], 'optional': []}
            }

def main():
    st.title("Scientific Review System")
    
    # Main configuration area
    st.markdown("## Document Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        doc_type = st.selectbox(
            "Document Type",
            ["Paper", "Grant", "Poster", "Presentation"],
            help="Select 'Presentation' for PowerPoint files"
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
    file_type_help = """
    - PDF files: Papers, Grants, and Posters
    - PowerPoint (.pptx): Presentations for slide-by-slide review
    """
    uploaded_file = st.file_uploader(
        "Upload Document", 
        type=["pdf", "pptx"],
        help=file_type_help
    )
    
    if uploaded_file:
        try:
            # Extract content with structure
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Auto-set document type based on file type
            if file_type == "pptx" and doc_type != "Presentation":
                st.info("Automatically setting document type to 'Presentation' for PowerPoint file.")
                doc_type = "Presentation"
                
            with st.spinner(f"Extracting {file_type.upper()} content..."):
                content, sections = extract_document_content(uploaded_file, file_type)
            
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
                        sections=sections,
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
