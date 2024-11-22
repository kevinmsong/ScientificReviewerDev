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
    """Display review results with improved formatting."""
    st.markdown("## Review Results")
    
    if not results.get('iterations'):
        st.warning("No review results available.")
        return
    
    # Create tabs for iterations
    tabs = st.tabs([f"Iteration {i+1}" for i in range(len(results['iterations']))])
    
    for idx, (tab, iteration) in enumerate(zip(tabs, results['iterations'])):
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
                        
                        st.markdown("### 🎯 Slide-by-Slide Review")
                        slide_tabs = st.tabs([f"Slide {sr['slide_number']}" for sr in review['slide_reviews']])
                        
                        for slide_tab, slide_review in zip(slide_tabs, review['slide_reviews']):
                            with slide_tab:
                                if slide_review.get('error'):
                                    st.error(slide_review['error'])
                                else:
                                    sections = parse_slide_review(slide_review['content'])
                                    
                                    for section_title, content in sections.items():
                                        st.markdown(f"#### {section_title}")
                                        st.markdown(content)
                                        st.markdown("---")
                                    
                                    if slide_review.get('scores'):
                                        st.markdown("#### Scores")
                                        for category, score in slide_review['scores'].items():
                                            st.markdown(f"**{category}:** {score}")
                        
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
    if 'moderator_summary' in results and results['moderator_summary']:
        st.markdown("## 🎯 Moderator Analysis")
        display_moderator_sections(results['moderator_summary'])

class ModeratorAgent:
    def __init__(self, model="gpt-4o"):
        self.model = ChatOpenAI(
            temperature=0.0,  # Keep moderator objective
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
            "Poster": "gpt-4-turbo-preview"  # Use GPT-4-turbo for posters
        }
        self.moderator = ModeratorAgent()
    
    def _calculate_temperature(self, bias: int) -> float:
        """Calculate temperature based on reviewer bias while keeping it within valid range."""
        # Base temperature of 0.7, adjusted by bias but kept within [0.0, 1.0]
        return max(0.0, min(1.0, 0.7 + (bias * 0.1)))
    
    def process_review(self, content: str, sections: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process document review with multiple iterations and moderation."""
        iterations = []
        all_reviews = []
        
        try:
            # Check if it's a PowerPoint file
            is_presentation = any(section.get('type') == 'slide' for section in sections)
            if is_presentation:
                config['doc_type'] = "Presentation"  # Override doc_type for presentations
            
            for iteration in range(config['iterations']):
                iteration_reviews = []
                previous_context = self._create_previous_context(all_reviews)
                
                for reviewer in config['reviewers']:
                    try:
                        temperature = self._calculate_temperature(config.get('bias', 0))
                        model = self.model_config.get(config['doc_type'], "o")
                        
                        if is_presentation:
                            presentation_reviewer = PresentationReviewer(model=model)
                            review_results = presentation_reviewer.review_presentation(
                                sections=sections,
                                expertise=reviewer['expertise']
                            )
                            
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
                        st.error(f"Error in review by {reviewer['expertise']}: {str(e)}")
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
            
            moderator_summary = self.moderator.summarize_discussion(iterations)
            
            return {
                'iterations': iterations,
                'moderator_summary': moderator_summary,
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'doc_type': config['doc_type'],
                'is_presentation': is_presentation
            }
            
        except Exception as e:
            st.error(f"Error in review process: {str(e)}")
            raise
    
    def _create_previous_context(self, reviews: List[Dict[str, Any]]) -> str:
        """Create context from previous reviews."""
        if not reviews:
            return ""
            
        context = "\nPrevious reviews:\n"
        for review in reviews:
            if not review.get('error', False):
                context += f"\n{review['reviewer']}:\n{review['content']}\n"
        return context

    def _process_sectioned_content(self, agent, sections: List[Dict[str, Any]], 
                                 reviewer: Dict[str, str], config: Dict[str, Any],
                                 iteration: int, previous_context: str) -> Dict[str, Any]:
        """Process content section by section (for posters and slides)."""
        try:
            if sections[0].get('type') == 'slide':
                # Use specialized presentation reviewer
                presentation_reviewer = PresentationReviewer(model=self.model_config[config['doc_type']])
                review_results = presentation_reviewer.review_presentation(
                    sections=sections,
                    expertise=reviewer['expertise']
                )
                
                return {
                    'reviewer': reviewer['expertise'],
                    'content': review_results['overall_review']['content'],
                    'slide_reviews': review_results['slide_reviews'],
                    'structure_analysis': review_results['structure_analysis'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Original section review logic for posters
                section_reviews = []
                for section in sections:
                    prompt = self._create_section_review_prompt(
                        doc_type=config['doc_type'],
                        expertise=reviewer['expertise'],
                        scoring_type=config['scoring'],
                        iteration=iteration,
                        previous_context=previous_context,
                        section_type=section['type'],
                        section_number=section.get('number', 1)
                    )
                    
                    response = agent.invoke([
                        HumanMessage(content=f"{prompt}\n\nContent:\n{section['content']}")
                    ])
                    section_reviews.append({
                        'section_number': section.get('number', 1),
                        'content': response.content
                    })
                
                # Compile section reviews into final review
                compilation_prompt = self._create_compilation_prompt(
                    doc_type=config['doc_type'],
                    expertise=reviewer['expertise'],
                    section_reviews=section_reviews
                )
                
                final_response = agent.invoke([HumanMessage(content=compilation_prompt)])
                
                return {
                    'reviewer': reviewer['expertise'],
                    'content': final_response.content,
                    'section_reviews': section_reviews,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logging.error(f"Error in sectioned content review: {e}")
            raise
    
    def _process_full_content(self, agent, content: str, reviewer: Dict[str, str],
                            config: Dict[str, Any], iteration: int,
                            previous_context: str) -> Dict[str, Any]:
        """Process full content for papers and grants."""
        prompt = self._create_review_prompt(
            doc_type=config['doc_type'],
            expertise=reviewer['expertise'],
            scoring_type=config['scoring'],
            iteration=iteration,
            previous_context=previous_context
        )
        
        response = agent.invoke([HumanMessage(content=f"{prompt}\n\nContent:\n{content}")])
        
        return {
            'reviewer': reviewer['expertise'],
            'content': response.content,
            'timestamp': datetime.now().isoformat()
        }

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
    """Specialized reviewer for PowerPoint presentations."""
    
    def __init__(self, model="gpt-4o"):
        self.model = model
    
    def review_presentation(self, sections: List[Dict[str, Any]], expertise: str) -> Dict[str, Any]:
        """Review a presentation slide by slide."""
        slide_reviews = []
        overall_structure = []
        
        # Review each slide
        for section in sections:
            if section['type'] == 'slide':
                review = self._review_slide(section, expertise)
                slide_reviews.append(review)
                overall_structure.append({
                    'slide_number': section.get('number', 1),
                    'content_type': self._identify_slide_type(section['content']),
                    'key_points': review.get('key_points', [])
                })
        
        # Generate overall presentation review
        overall_review = self._generate_overall_review(slide_reviews, overall_structure, expertise)
        
        return {
            'slide_reviews': slide_reviews,
            'overall_review': overall_review,
            'structure_analysis': overall_structure
        }
    
    def _review_slide(self, slide: Dict[str, Any], expertise: str) -> Dict[str, Any]:
        """Review an individual slide."""
        prompt = f"""As a {expertise}, review this presentation slide:

Slide {slide.get('number', 1)}:
{slide['content']}

Provide a structured analysis:

1. SLIDE PURPOSE:
   - Main message/objective
   - Target audience relevance
   - Position in presentation flow

2. CONTENT ANALYSIS:
   - Key points
   - Technical accuracy
   - Data presentation (if applicable)
   - Supporting evidence

3. VISUAL DESIGN:
   - Layout effectiveness
   - Visual hierarchy
   - Use of space
   - Graphics and images

4. COMMUNICATION:
   - Clarity of message
   - Text conciseness
   - Technical level appropriateness

5. RECOMMENDATIONS:
   [REQUIRED] Critical improvements needed
   [OPTIONAL] Enhancement suggestions

6. SLIDE SCORE (1-5 stars):
   - Content Quality: ★☆☆☆☆
   - Visual Design: ★☆☆☆☆
   - Communication Effectiveness: ★☆☆☆☆"""

        try:
            agent = ChatOpenAI(
                temperature=0.1,
                openai_api_key=st.secrets["openai_api_key"],
                model=self.model
            )
            response = agent.invoke([HumanMessage(content=prompt)])
            
            # Parse the review response
            review = {
                'slide_number': slide.get('number', 1),
                'content': response.content,
                'scores': self._extract_scores(response.content),
                'key_points': self._extract_key_points(response.content),
                'recommendations': self._extract_recommendations(response.content)
            }
            
            return review
        except Exception as e:
            logging.error(f"Error reviewing slide {slide.get('number', 1)}: {e}")
            return {
                'slide_number': slide.get('number', 1),
                'error': str(e)
            }
    
    def _generate_overall_review(self, slide_reviews: List[Dict[str, Any]], 
                               structure: List[Dict[str, Any]], expertise: str) -> Dict[str, Any]:
        """Generate overall presentation review."""
        prompt = f"""As a {expertise}, synthesize this complete presentation review:

Structure Summary:
{json.dumps(structure, indent=2)}

Individual Slide Reviews:
{json.dumps([r.get('content', '') for r in slide_reviews], indent=2)}

Provide a comprehensive review:

1. PRESENTATION OVERVIEW:
   - Main objectives
   - Target audience fit
   - Story/flow effectiveness

2. STRUCTURE ANALYSIS:
   - Logical flow
   - Section transitions
   - Time allocation
   - Content balance

3. CONTENT QUALITY:
   - Technical depth
   - Evidence support
   - Data presentation
   - Message clarity

4. VISUAL CONSISTENCY:
   - Design template
   - Color scheme
   - Typography
   - Branding elements

5. KEY STRENGTHS AND WEAKNESSES:
   - Major effective elements
   - Critical improvement areas

6. RECOMMENDATIONS:
   [REQUIRED] Essential improvements
   [OPTIONAL] Enhancement suggestions

7. OVERALL SCORES (1-5 stars):
   - Content Organization: ★☆☆☆☆
   - Visual Design: ★☆☆☆☆
   - Technical Quality: ★☆☆☆☆
   - Presentation Impact: ★☆☆☆☆"""

        try:
            agent = ChatOpenAI(
                temperature=0.1,
                openai_api_key=st.secrets["openai_api_key"],
                model=self.model
            )
            response = agent.invoke([HumanMessage(content=prompt)])
            
            return {
                'content': response.content,
                'scores': self._extract_scores(response.content),
                'recommendations': self._extract_recommendations(response.content)
            }
        except Exception as e:
            logging.error(f"Error generating overall review: {e}")
            return {'error': str(e)}
    
    def _identify_slide_type(self, content: str) -> str:
        """Identify the type/purpose of a slide."""
        slide_types = {
            'title': ['agenda', 'outline', 'overview', 'content'],
            'introduction': ['introduction', 'background', 'context'],
            'methods': ['methods', 'methodology', 'approach', 'procedure'],
            'results': ['results', 'findings', 'data', 'analysis'],
            'discussion': ['discussion', 'implications', 'interpretation'],
            'conclusion': ['conclusion', 'summary', 'takeaways'],
            'references': ['references', 'citations', 'bibliography']
        }
        
        content_lower = content.lower()
        for slide_type, keywords in slide_types.items():
            if any(keyword in content_lower for keyword in keywords):
                return slide_type
        return 'content'  # default type
    
    def _extract_scores(self, content: str) -> Dict[str, int]:
        """Extract numerical scores from review content."""
        scores = {}
        score_pattern = r'(\w+(?:\s+\w+)*)\s*:\s*([★]+(?:☆)*)'
        
        matches = re.finditer(score_pattern, content)
        for match in matches:
            category = match.group(1).strip()
            stars = match.group(2)
            scores[category] = stars.count('★')
        
        return scores
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from review content."""
        points = []
        if 'Key points' in content:
            section = content.split('Key points')[1].split('\n\n')[0]
            points = [point.strip('- ').strip() for point in section.split('\n') 
                     if point.strip('- ').strip()]
        return points
    
    def _extract_recommendations(self, content: str) -> Dict[str, List[str]]:
        """Extract recommendations from review content."""
        recommendations = {'required': [], 'optional': []}
        
        if '[REQUIRED]' in content:
            required_section = content.split('[REQUIRED]')[1].split('[OPTIONAL]')[0]
            recommendations['required'] = [rec.strip('- ').strip() 
                                        for rec in required_section.split('\n') 
                                        if rec.strip('- ').strip()]
        
        if '[OPTIONAL]' in content:
            optional_section = content.split('[OPTIONAL]')[1].split('\n\n')[0]
            recommendations['optional'] = [rec.strip('- ').strip() 
                                        for rec in optional_section.split('\n') 
                                        if rec.strip('- ').strip()]
        
        return recommendations

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
