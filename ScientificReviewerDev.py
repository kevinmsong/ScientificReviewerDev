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
import time

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

def clean_text_formatting(text: str) -> str:
    """Clean text formatting with enhanced number removal."""
    # Remove asterisks not part of bullet points
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Remove numbered bullets
    text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)
    # Remove numbered bullets with asterisks
    text = re.sub(r'\d+\.\s*\*\*', '', text)
    # Remove lone asterisks
    text = re.sub(r'^\*\s*', '• ', text, flags=re.MULTILINE)
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Clean up bullet points
    text = re.sub(r'•\s*\*\*([^*]+)\*\*:', r'• \1:', text)
    # Remove any remaining numbered lists
    text = re.sub(r'^\d+\)\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def extract_pptx_content(pptx_file) -> tuple[str, List[Dict[str, Any]]]:
    """Extract content from PowerPoint with slide preservation and improved error handling."""
    temp_file = "temp.pptx"
    try:
        if not pptx_file or not pptx_file.getvalue():
            raise ValueError("Empty or invalid PowerPoint file")

        # Save uploaded file temporarily
        with open(temp_file, "wb") as f:
            f.write(pptx_file.getvalue())
        
        # Load and process slides
        loader = UnstructuredPowerPointLoader(temp_file)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content found in PowerPoint file")

        full_content = ""
        sections = []
        
        for idx, doc in enumerate(documents):
            content = doc.page_content
            if not content.strip():
                logging.warning(f"Empty content in slide {idx + 1}")
                continue

            full_content += f"\nSlide {idx + 1}:\n{content}\n"
            sections.append({
                'type': 'slide',
                'number': idx + 1,
                'content': content,
                'metadata': doc.metadata
            })
        
        if not sections:
            raise ValueError("No valid slides found in presentation")

        return full_content.strip(), sections
        
    except Exception as e:
        error_msg = f"Error processing PowerPoint: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)
        
    finally:
        # Clean up temporary file
        import os
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                logging.error(f"Failed to remove temporary file: {str(e)}")

def parse_review_sections(content: str) -> Dict[str, str]:
    """Parse review content into structured sections with enhanced formatting."""
    sections = {}
    
    # Define section markers and their keys
    section_markers = {
        'RESPONSE TO PREVIOUS REVIEWS': 'response',
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
            
            section_content = content[start + len(marker):next_section_start].strip()
            
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

def display_review_sections(sections: Dict[str, str], is_nih: bool = False):
    """Display review sections with enhanced formatting."""
    section_order = [
        'response', 'significance', 'innovation', 
        'approach', 'scoring', 'recommendations'
    ]
    
    icons = {
        'response': '💬',
        'significance': '🎯',
        'innovation': '💡',
        'approach': '🔍',
        'scoring': '⭐',
        'recommendations': '📋'
    }
    
    titles = {
        'response': 'Response to Previous Reviews',
        'significance': 'Significance Evaluation',
        'innovation': 'Innovation Assessment',
        'approach': 'Approach Analysis',
        'scoring': 'Scoring',
        'recommendations': 'Recommendations'
    }
    
    for section in section_order:
        if section in sections:
            content = sections[section]
            
            st.markdown(f"### {icons[section]} {titles[section]}")
            
            if is_nih and section in ['significance', 'innovation', 'approach']:
                # Format NIH-specific sections
                lines = content.split('\n')
                current_subsection = None
                subsection_content = []
                
                for line in lines:
                    if re.match(r'^\d+\.', line):
                        if current_subsection and subsection_content:
                            st.markdown(f"**{current_subsection}**")
                            for point in subsection_content:
                                if point.strip():
                                    st.markdown(f"- {point.strip()}")
                            st.markdown("---")
                        current_subsection = line.strip()
                        subsection_content = []
                    elif line.strip():
                        subsection_content.append(line)
                
                if current_subsection and subsection_content:
                    st.markdown(f"**{current_subsection}**")
                    for point in subsection_content:
                        if point.strip():
                            st.markdown(f"- {point.strip()}")
                    st.markdown("---")
                
            elif section == 'scoring':
                # Format scoring section
                scores = re.findall(r'([^:]+):\s*\[(\d+)\][^\n]*(?:\n|$)', content)
                for category, score in scores:
                    category = category.strip()
                    if is_nih:
                        st.markdown(f"**{category}:** {score}/9")
                        # Extract and display score justification
                        justification = re.search(f"{category}:.*?\\[{score}\\](.*?)(?=\\n\\w|$)", content, re.DOTALL)
                        if justification:
                            st.markdown(justification.group(1).strip())
                    else:
                        st.markdown(f"**{category}:** {'★' * int(score)}{'☆' * (5 - int(score))}")
                st.markdown("---")
                
            elif section == 'recommendations':
                # Format recommendations section
                required = []
                optional = []
                current_list = None
                
                for line in content.split('\n'):
                    if 'Critical changes needed:' in line:
                        current_list = required
                    elif 'Suggested improvements:' in line:
                        current_list = optional
                    elif line.strip().startswith('-') and current_list is not None:
                        current_list.append(line.strip()[1:].strip())
                
                if required:
                    st.markdown("**Required Changes:**")
                    for item in required:
                        st.markdown(f"- {item}")
                
                if optional:
                    st.markdown("**Optional Improvements:**")
                    for item in optional:
                        st.markdown(f"- {item}")
                st.markdown("---")
                
            else:
                # Format regular sections
                paragraphs = content.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        st.markdown(paragraph.strip())
                st.markdown("---")

def format_analysis_section(title: str, content: str) -> str:
    """Format a section of the analysis with clean styling."""
    formatted_content = clean_text_formatting(content)
    
    # Split into bullet points if content contains them
    if '• ' in formatted_content:
        points = formatted_content.split('• ')
        formatted_points = []
        for point in points:
            if point.strip():
                if ':' in point:
                    label, detail = point.split(':', 1)
                    formatted_points.append(f"• {label.strip()}:{detail}")
                else:
                    formatted_points.append(f"• {point.strip()}")
        formatted_content = '\n'.join(formatted_points)
    
    return f"{title}\n{formatted_content}\n"

def format_dialogue(dialogue_content: str) -> str:
    """Format reviewer dialogue with clean styling."""
    lines = dialogue_content.split('\n')
    formatted_lines = []
    
    for line in lines:
        if ':' in line and not line.strip().endswith(':'):
            speaker, message = line.split(':', 1)
            if re.match(r'^[A-Za-z\s]+$', speaker.strip()):
                formatted_lines.append(f"{speaker.strip()}: {message.strip()}")
            else:
                formatted_lines.append(line.strip())
        elif line.strip():
            formatted_lines.append(line.strip())
    
    return '\n\n'.join(formatted_lines)

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

def parse_nih_review_sections(content: str) -> Dict[str, str]:
    """Parse NIH grant review content with special formatting."""
    sections = {}
    
    section_markers = {
        'RESPONSE TO PREVIOUS REVIEWS': 'response',
        'SIGNIFICANCE EVALUATION': 'significance',
        'INNOVATION ASSESSMENT': 'innovation',
        'APPROACH ANALYSIS': 'approach',
        'SCORING': 'scoring',
        'RECOMMENDATIONS': 'recommendations'
    }
    
    subsection_patterns = {
        'significance': [
            r'1\.\s*Important\s*Problem/Critical\s*Barrier:(.*?)(?=2\.|$)',
            r'2\.\s*Scientific/Technical/Clinical\s*Advancement:(.*?)(?=3\.|$)',
            r'3\.\s*Impact\s*Potential:(.*?)(?=\n\n[A-Z]|$)'
        ],
        'innovation': [
            r'1\.\s*Research\s*Paradigm\s*Challenge:(.*?)(?=2\.|$)',
            r'2\.\s*Novelty\s*Analysis:(.*?)(?=3\.|$)',
            r'3\.\s*State-of-the-Art\s*Advancement:(.*?)(?=\n\n[A-Z]|$)'
        ],
        'approach': [
            r'1\.\s*Strategy\s*and\s*Methodology:(.*?)(?=2\.|$)',
            r'2\.\s*Risk\s*Management:(.*?)(?=3\.|$)',
            r'3\.\s*Feasibility:(.*?)(?=\n\n[A-Z]|$)'
        ]
    }
    
    # Extract main sections
    for marker, key in section_markers.items():
        if marker in content:
            start = content.find(marker)
            next_section_start = float('inf')
            for other_marker in section_markers:
                if other_marker != marker:
                    pos = content.find(other_marker, start + len(marker))
                    if pos != -1 and pos < next_section_start:
                        next_section_start = pos
            
            if next_section_start == float('inf'):
                next_section_start = len(content)
            
            section_content = content[start + len(marker):next_section_start].strip()
            
            # Format subsections for NIH-specific sections
            if key in subsection_patterns:
                formatted_content = ""
                for pattern in subsection_patterns[key]:
                    match = re.search(pattern, section_content, re.DOTALL | re.IGNORECASE)
                    if match:
                        subsection_content = match.group(1).strip()
                        formatted_content += f"{match.group(0).split(':')[0]}:\n{subsection_content}\n\n"
                sections[key] = formatted_content.strip()
            else:
                sections[key] = section_content
    
    return sections

def display_review_results(results: Dict[str, Any]):
    """Display review results with enhanced formatting."""
    try:
        # [Previous validation code remains the same...]
        
        # Display final analysis in its own tab
        with tabs[-1]:
            try:
                if results.get('moderator_summary'):
                    if is_nih:
                        st.markdown("### NIH Review Summary")
                    
                    # Parse and display moderator sections without numbering
                    sections = parse_moderator_sections(results['moderator_summary'])
                    
                    # Display each section with clean formatting and no numbers
                    section_titles = {
                        'agreement': '🤝 Points of Agreement',
                        'contention': '⚖️ Points of Contention',
                        'evolution': '📈 Discussion Evolution',
                        'synthesis': '🎯 Final Synthesis'
                    }
                    
                    for key, title in section_titles.items():
                        if key in sections:
                            st.markdown(f"### {title}")
                            # Clean formatting and ensure no numbered lists
                            formatted_content = clean_text_formatting(sections[key])
                            # Remove any remaining numbered points
                            formatted_content = re.sub(r'^\d+\.\s+', '', formatted_content, flags=re.MULTILINE)
                            st.markdown(formatted_content)
                            st.markdown("---")
                else:
                    st.warning("No final analysis available.")
            except Exception as e:
                st.error(f"Error displaying final analysis: {str(e)}")
                if results.get('moderator_summary'):
                    st.markdown(results['moderator_summary'])

    except Exception as e:
        st.error(f"Error displaying review results: {str(e)}")
        logging.error(f"Review display error: {str(e)}")

class ModeratorAgent:
    def __init__(self, model="gpt-4o"):
        self.model = ChatOpenAI(
            temperature=0.0,
            openai_api_key=st.secrets["openai_api_key"],
            model=model
        )
    
    def summarize_discussion(self, iterations: List[Dict[str, Any]]) -> str:
        """Analyze the complete discussion history across all iterations."""
        try:
            if not iterations:
                return "No iterations to analyze."

            discussion_summary = "Complete Discussion History:\n\n"
            
            # Build comprehensive discussion history
            for idx, iteration in enumerate(iterations, 1):
                discussion_summary += f"\nIteration {idx}:\n"
                
                # Add reviews
                if iteration.get('reviews'):
                    for review in iteration['reviews']:
                        if not review.get('error', False):
                            reviewer = review.get('reviewer', 'Unknown Reviewer')
                            content = review.get('content', 'No content available')
                            discussion_summary += f"\n{reviewer}:\n{content}\n"
                
                # Add dialogue and key points
                if iteration.get('dialogue'):
                    discussion_summary += "\nDiscussion Summary:\n"
                    for dialogue in iteration['dialogue']:
                        if isinstance(dialogue, dict):
                            if dialogue.get('key_points'):
                                discussion_summary += f"{dialogue['key_points']}\n"
                            if dialogue.get('content'):
                                discussion_summary += f"Detailed Discussion:\n{dialogue['content']}\n"

            moderator_prompt = f"""As a moderator, analyze this complete discussion history across all iterations:

{discussion_summary}

Provide a comprehensive final analysis using the following structure. Do not use numbered sections or bullet points - use clear paragraph breaks and descriptive text instead:

KEY POINTS OF AGREEMENT
Focus on how consensus was built across iterations. Describe major agreements and their evolution. Indicate when key agreements were reached during the discussion. Present this as a flowing narrative rather than a list.

POINTS OF CONTENTION
Analyze the unresolved disagreements and examine how they evolved or were resolved. Identify root causes of persistent disagreements. Describe the different perspectives and why they remain unreconciled.

DISCUSSION EVOLUTION
Track how viewpoints changed across iterations. Describe key turning points in the discussion and what influenced opinion changes. Analyze how effectively the dialogue progressed and how perspectives shifted over time.

FINAL SYNTHESIS
Provide a cohesive summary of the overall consensus view. Present critical recommendations that emerged from the discussion. Outline clear next steps and areas requiring further attention. Note any remaining uncertainties and suggest future directions.

Format your response using these exact section headings but present the content in a narrative style without numbering or bullet points."""

            try:
                response = self.model.invoke([HumanMessage(content=moderator_prompt)])
                if not response or not hasattr(response, 'content'):
                    raise ValueError("Invalid model response")
                return response.content
            except Exception as e:
                logging.error(f"Error in moderation: {e}")
                return f"Error generating moderation summary: {str(e)}"

        except Exception as e:
            logging.error(f"Error in discussion summary: {e}")
            return f"Error analyzing discussion: {str(e)}"

class PresentationReviewer:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = None
    
    def _generate_overall_review(self, slide_reviews: List[Dict[str, Any]], structure: List[Dict[str, Any]], expertise: str) -> Dict[str, Any]:
        """Generate overall presentation review."""
        try:
            if not slide_reviews:
                raise ValueError("No slide reviews to analyze")

            structure_summary = "\n".join([
                f"Slide {s['slide_number']}: {s['content_type'].title()}"
                for s in structure
            ])

            # Create a summary of slide reviews for the prompt
            reviews_summary = []
            for review in slide_reviews:
                review_summary = {
                    'slide_number': review['slide_number'],
                    'key_points': review.get('key_points', []),
                    'scores': review.get('scores', {}),
                    'recommendations': review.get('recommendations', {'required': [], 'optional': []})
                }
                reviews_summary.append(review_summary)

            prompt = f"""As a {expertise}, synthesize this complete presentation review:

Structure Overview:
{structure_summary}

Individual Slide Reviews:
{json.dumps(reviews_summary, indent=2)}

Provide a comprehensive review using these exact sections:

OVERALL ASSESSMENT:
- Main objectives and target audience
- Flow and structure effectiveness
- Technical depth and clarity
- Overall presentation quality

KEY STRENGTHS:
- Major effective elements
- Successful aspects
- Notable achievements

CRITICAL IMPROVEMENTS:
- Essential changes needed
- Major issues to address
- Priority fixes

RECOMMENDATIONS:
[REQUIRED] Essential changes needed:
- List specific required improvements
- Focus on critical fixes

[OPTIONAL] Enhancement suggestions:
- List potential improvements
- Additional refinements

OVERALL SCORES (using ★):
Content Organization: (1-5 stars)
Visual Design: (1-5 stars)
Technical Quality: (1-5 stars)
Presentation Impact: (1-5 stars)"""

            response = self.client.invoke([HumanMessage(content=prompt)])
            
            if not response or not hasattr(response, 'content'):
                raise ValueError("Invalid AI response for overall review")

            content = response.content

            return {
                'content': content,
                'scores': self._extract_scores(content),
                'recommendations': self._extract_recommendations(content)
            }

        except Exception as e:
            error_msg = f"Overall review generation failed: {str(e)}"
            logging.error(error_msg)
            return {
                'content': error_msg,
                'scores': {},
                'recommendations': {'required': [], 'optional': []}
            }
    
    def _review_slide(self, slide: Dict[str, Any], expertise: str) -> Dict[str, Any]:
        """Review an individual slide with relaxed validation."""
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
            
            # Create review with all components, allowing empty results
            review = {
                'slide_number': slide.get('number', 1),
                'content': content,
                'scores': self._extract_scores(content) or {},  # Allow empty scores
                'key_points': self._extract_key_points(content) or [],  # Allow empty points
                'recommendations': self._extract_recommendations(content) or {'required': [], 'optional': []}
            }

            # Basic validation - only check if we have content
            if not review['content']:
                raise ValueError("Empty review content")

            return review

        except Exception as e:
            error_msg = f"Slide review failed: {str(e)}"
            logging.error(error_msg)
            return {
                'slide_number': slide.get('number', 1),
                'error': True,
                'content': f"Review failed for slide {slide.get('number', 1)}: {str(e)}"
            }

    def review_presentation(self, sections: List[Dict[str, Any]], expertise: str, temperature: float = 0.7) -> Dict[str, Any]:
        """Review a presentation slide by slide."""
        try:
            if not sections:
                raise ValueError("No sections provided for review")

            # Initialize OpenAI client
            self.client = ChatOpenAI(
                temperature=temperature,
                openai_api_key=st.secrets["openai_api_key"],
                model=self.model
            )

            # Filter and sort slides
            slide_sections = [s for s in sections if s.get('type') == 'slide']
            print(f"Found {len(slide_sections)} slides to review")  # Debug print

            slide_reviews = []
            overall_structure = []

            # Process each slide with explicit numbering
            for slide_num, section in enumerate(slide_sections, 1):
                print(f"Processing slide {slide_num}")  # Debug print
                try:
                    if not section.get('content'):
                        logging.warning(f"Skipping slide {slide_num} - no content")
                        continue

                    # Ensure the slide has a number
                    section['number'] = slide_num
                    review = self._review_slide(section, expertise)
                    
                    if review.get('content') and not review.get('error'):
                        formatted_content = self._format_review_text(review['content'])
                        review['content'] = formatted_content
                        review['slide_number'] = slide_num  # Ensure slide number is set
                        print(f"Successfully reviewed slide {slide_num}")  # Debug print
                        slide_reviews.append(review)
                        
                        overall_structure.append({
                            'slide_number': slide_num,
                            'content_type': self._identify_slide_type(section['content']),
                            'key_points': review.get('key_points', [])
                        })
                    else:
                        error_msg = f"Error in slide {slide_num}: {review.get('error', 'Unknown error')}"
                        logging.error(error_msg)
                        print(error_msg)  # Debug print

                except Exception as e:
                    error_msg = f"Error processing slide {slide_num}: {str(e)}"
                    logging.error(error_msg)
                    print(error_msg)  # Debug print

            # Verify we processed all slides
            print(f"Processed {len(slide_reviews)} slide reviews")  # Debug print

            if not slide_reviews:
                raise ValueError("No valid slides were processed")

            # Sort reviews by slide number to ensure correct order
            slide_reviews.sort(key=lambda x: x.get('slide_number', float('inf')))

            # Generate overall review
            overall_review = self._generate_overall_review(slide_reviews, overall_structure, expertise)
            if overall_review and overall_review.get('content'):
                overall_review['content'] = self._format_review_text(overall_review['content'])
            else:
                overall_review = {
                    'content': "Failed to generate overall review",
                    'scores': {},
                    'recommendations': {'required': [], 'optional': []}
                }

            result = {
                'slide_reviews': slide_reviews,
                'overall_review': overall_review,
                'structure_analysis': overall_structure
            }

            # Debug print final structure
            print(f"Final result has {len(result['slide_reviews'])} slide reviews")
            for review in result['slide_reviews']:
                print(f"Review for slide {review.get('slide_number')}")

            return result

        except Exception as e:
            error_msg = f"Presentation review failed: {str(e)}"
            logging.error(error_msg)
            print(error_msg)  # Debug print
            return {
                'error': True,
                'message': error_msg,
                'slide_reviews': [],
                'overall_review': {'content': error_msg},
                'structure_analysis': []
            }

    def _extract_scores(self, content: str) -> Dict[str, int]:
        """Extract numerical scores from review content."""
        scores = {}
        try:
            patterns = [
                r'([^:\n]+?):\s*(★+(?:☆)*)',
                r'([^:\n]+?):\s*(\d+)(?:\s*stars?)?(?=[\s\n]|$)',
                r'([^:\n]+?):\s*(\d+)/5'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    category = match.group(1).strip()
                    score_text = match.group(2).strip()
                    
                    if '★' in score_text:
                        score = score_text.count('★')
                    else:
                        try:
                            score = int(float(score_text))
                        except ValueError:
                            continue
                    
                    score = max(1, min(5, score))
                    if category and score:
                        scores[category] = score
            
            return scores
        except Exception as e:
            logging.error(f"Error extracting scores: {str(e)}")
            return {}

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from review content."""
        key_points = []
        try:
            sections = [
                'CONTENT ANALYSIS',
                'KEY POINTS',
                'MAIN POINTS',
                'SLIDE PURPOSE'
            ]
            
            for section in sections:
                pattern = fr'{section}:?(.*?)(?=\n\n[A-Z\s]+:|$)'
                matches = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if matches:
                    section_text = matches.group(1).strip()
                    points = re.findall(r'(?:^|\n)\s*[-•*]\s*(.*?)(?=\n|$)', section_text)
                    if not points:
                        points = re.findall(r'(?:^|\n)\s*\d+\.\s*(.*?)(?=\n|$)', section_text)
                    
                    points = [p.strip() for p in points if p.strip()]
                    if points:
                        key_points.extend(points)
                        break
            
            if not key_points:
                # Fallback to extracting sentences
                sentences = re.split(r'[.!?]+\s+', content)
                key_points = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
            
            return key_points[:5]  # Limit to top 5 points
            
        except Exception as e:
            logging.error(f"Error extracting key points: {str(e)}")
            return []

    def _extract_recommendations(self, content: str) -> Dict[str, List[str]]:
        """Extract recommendations from review content."""
        recommendations = {'required': [], 'optional': []}
        try:
            required_patterns = [
                r'\[REQUIRED\](.*?)(?=\[OPTIONAL\]|\n\n[A-Z]|\Z)',
                r'Critical improvements needed:(.*?)(?=\n\n[A-Z]|\Z)',
                r'Essential improvements:(.*?)(?=\n\n[A-Z]|\Z)'
            ]
            
            optional_patterns = [
                r'\[OPTIONAL\](.*?)(?=\n\n[A-Z]|\Z)',
                r'Enhancement suggestions?:(.*?)(?=\n\n[A-Z]|\Z)',
                r'Suggested improvements:(.*?)(?=\n\n[A-Z]|\Z)'
            ]
            
            def extract_points(patterns: List[str], content: str) -> List[str]:
                for pattern in patterns:
                    matches = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    if matches:
                        section_text = matches.group(1).strip()
                        points = re.findall(r'(?:^|\n)\s*[-•*]\s*(.*?)(?=\n|$)', section_text)
                        if not points:
                            points = re.findall(r'(?:^|\n)\s*\d+\.\s*(.*?)(?=\n|$)', section_text)
                        points = [p.strip() for p in points if p.strip()]
                        if points:
                            return points
                return []
            
            recommendations['required'] = extract_points(required_patterns, content)
            recommendations['optional'] = extract_points(optional_patterns, content)
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error extracting recommendations: {str(e)}")
            return {'required': [], 'optional': []}

    def _format_review_text(self, raw_content: str) -> str:
        """Format and clean review text for better presentation."""
        try:
            # Remove excessive newlines
            content = re.sub(r'\n{3,}', '\n\n', raw_content)
            
            # Format section headers
            content = re.sub(r'(?m)^([A-Z][A-Z\s]+):', r'\n\1:', content)
            
            # Clean up bullet points
            content = re.sub(r'(?m)^\s*[*•]\s*', '• ', content)
            
            # Format star ratings
            content = re.sub(r'(\d)\s*stars?\b', r'\1★', content)
            content = re.sub(r'(\d)/5', r'\1★', content)
            content = content.replace('*', '★')
            
            return content.strip()
            
        except Exception as e:
            logging.error(f"Error formatting review text: {str(e)}")
            return raw_content

    def _identify_slide_type(self, content: str) -> str:
        """Identify the type/purpose of a slide."""
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
        return 'content'

class ReviewManager:
    def __init__(self):
        self.model_config = {
            "Paper": "gpt-4o",
            "Grant": "gpt-4o",
            "Poster": "gpt-4o",
            "Presentation": "gpt-4o"
        }
        self.moderator = ModeratorAgent()

    def _generate_reviewer_dialogue(self, reviews: List[Dict[str, Any]], previous_iterations: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a dialogue between reviewers that builds on previous discussions."""
        try:
            if not reviews:
                raise ValueError("No reviews provided for dialogue generation")
                
            valid_reviews = [r for r in reviews if (
                isinstance(r, dict) and 
                not r.get('error', False) and 
                r.get('reviewer') and 
                r.get('content')
            )]
            
            if not valid_reviews:
                raise ValueError("No valid reviews found for dialogue generation")

            current_iteration = len(previous_iterations) + 1
            dialogue_context = ""
            
            # Build context from previous iterations
            if previous_iterations:
                dialogue_context = "Previous Discussions:\n\n"
                for idx, iteration in enumerate(previous_iterations, 1):
                    dialogue_context += f"Iteration {idx} Key Points:\n"
                    if iteration.get('dialogue'):
                        for dialogue in iteration['dialogue']:
                            if isinstance(dialogue, dict) and dialogue.get('key_points'):
                                dialogue_context += f"{dialogue['key_points']}\n"
                    dialogue_context += "\n"

            # Extract reviewer information
            reviewers = [review['reviewer'] for review in valid_reviews]
            
            # Create dialogue prompt
            dialogue_prompt = f"""Based on the following {len(reviewers)} expert reviews and previous discussions, generate a detailed dialogue between the reviewers. This is iteration {current_iteration}.

    Reviewers: {', '.join(reviewers)}

    Previous Context:
    {dialogue_context}

    Current Reviews:
    """
            # Add review content
            for review in valid_reviews:
                dialogue_prompt += f"\n{review['reviewer']}:\n{review['content']}\n"

            dialogue_prompt += """
    Generate a natural dialogue that:
    1. References and builds upon previous discussions and agreements
    2. Addresses any unresolved points from earlier iterations
    3. Works toward consensus where possible
    4. Maintains individual expertise perspectives
    5. Considers both technical and broader impacts

    Track key points of:
    - Agreements reached
    - Remaining disagreements
    - New insights or perspectives
    - Action items or recommendations

    Format as a detailed conversation with clear speaker labels and logical progression."""

            # Generate dialogue
            agent = ChatOpenAI(
                temperature=0.7,
                openai_api_key=st.secrets["openai_api_key"],
                model="gpt-4o"
            )
            
            response = agent.invoke([HumanMessage(content=dialogue_prompt)])
            
            # Generate summary of key points
            summary_prompt = f"""Based on this dialogue:
    {response.content}

    Provide a concise summary of:
    1. Key agreements reached
    2. Remaining points of contention
    3. New insights or perspectives gained
    4. Specific recommendations or action items

    Focus on how this discussion evolved from previous iterations and what progress was made."""

            summary_response = agent.invoke([HumanMessage(content=summary_prompt)])
            
            return {
                'content': response.content,
                'key_points': summary_response.content,
                'timestamp': datetime.now().isoformat(),
                'reviewers': reviewers,
                'iteration': current_iteration
            }
            
        except Exception as e:
            error_msg = f"Dialogue generation failed: {str(e)}"
            logging.error(error_msg)
            return {
                'content': error_msg,
                'key_points': '',
                'timestamp': datetime.now().isoformat(),
                'error': True,
                'reviewers': [],
                'iteration': len(previous_iterations) + 1
            }

    def _process_full_content(self, agent, content: str, reviewer: Dict[str, Any], config: Dict[str, Any], iteration: int, previous_context: str) -> Dict[str, Any]:
        """Process a full document review with structured sections."""
        try:
            # Special handling for NIH grants
            if config['doc_type'] == "Grant" and config['scoring'] == "nih":
                prompt = f"""As a {reviewer['expertise']}, review this NIH grant for {config['venue']}.

    Iteration: {iteration}
    {previous_context}

    Document Content:
    {content}

    Provide a detailed review using these exact sections:

    RESPONSE TO PREVIOUS REVIEWS:
    Only address previous reviews if this is a later iteration.

    SIGNIFICANCE EVALUATION:
    1. Important Problem/Critical Barrier:
    - Does the project address an important problem or critical barrier to progress in the field?
    - Specific evidence and analysis required

    2. Scientific/Technical/Clinical Advancement:
    - How will scientific knowledge, technical capability, or clinical practice be advanced?
    - Concrete examples of potential advances

    3. Impact Potential:
    - What is the potential to change concepts, methods, technologies, or treatments?
    - Specific areas of impact

    INNOVATION ASSESSMENT:
    1. Research Paradigm Challenge:
    - Does the project challenge or seek to shift current research paradigms?
    - Specific examples of paradigm shifts

    2. Novelty Analysis:
    - Are concepts, approaches, or methodologies novel?
    - Detailed assessment of originality

    3. State-of-the-Art Advancement:
    - Is the work a refinement, improvement, or application of state-of-the-art ideas?
    - Specific technological or methodological advances

    APPROACH ANALYSIS:
    1. Strategy and Methodology:
    - Are the strategy, methodology, and analyses well-reasoned and appropriate?
    - Detailed assessment of research plan

    2. Risk Management:
    - Are potential problems, alternative strategies, and benchmarks for success considered?
    - Specific contingency plans

    3. Feasibility:
    - Is the approach feasible within the proposed timeline and resources?
    - Resource and timeline analysis

    SCORING (NIH Scale 1-9, where 1 is exceptional and 9 is poor):
    - Significance Score: [score]
    Detailed justification...
    - Innovation Score: [score]
    Detailed justification...
    - Approach Score: [score]
    Detailed justification...
    - Overall Impact Score: [score]
    Detailed justification...

    RECOMMENDATIONS:
    Critical changes needed:
    - List required revisions

    Suggested improvements:
    - List optional enhancements"""
            else:
                # Original prompt for other document types
                scoring_guide = (
                    "Rate each section 1-5 stars (★)" if config['scoring'] == 'stars'
                    else "Rate each section on NIH scale (1-9, where 1 is exceptional and 9 is poor)"
                )
                
                prompt = f"""As a {reviewer['expertise']}, review this {config['doc_type'].lower()} for {config['venue']}.

    Iteration: {iteration}
    {previous_context}

    Document Content:
    {content}

    Provide a detailed review using these exact sections:

    RESPONSE TO PREVIOUS REVIEWS:
    Only address previous reviews if this is a later iteration.

    SECTION-BY-SECTION ANALYSIS:
    For each major section:
    - Content Summary
    - Critical Changes
    - Suggested Improvements
    - Specific Line Edits

    SIGNIFICANCE EVALUATION:
    - Scientific impact
    - Clinical/translational relevance
    - Innovation potential
    - Contribution to field

    INNOVATION ASSESSMENT:
    - Novel concepts/approaches
    - Technical innovations
    - Methodological advances
    - Potential impact

    APPROACH ANALYSIS:
    - Technical feasibility
    - Methodology appropriateness
    - Statistical considerations
    - Potential pitfalls and alternatives

    SCORING:
    {scoring_guide}
    - Significance: [score]
    - Innovation: [score]
    - Approach: [score]
    - Overall Impact: [score]

    RECOMMENDATIONS:
    Critical changes needed:
    - List required revisions

    Suggested improvements:
    - List optional enhancements"""

            response = agent.invoke([HumanMessage(content=prompt)])
            
            if not response or not hasattr(response, 'content'):
                raise ValueError("Invalid AI response")

            return {
                'reviewer': reviewer['expertise'],
                'content': response.content,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Review generation failed: {str(e)}"
            logging.error(error_msg)
            return {
                'reviewer': reviewer['expertise'],
                'content': error_msg,
                'timestamp': datetime.now().isoformat(),
                'error': True
            }

    def _calculate_temperature(self, bias: int) -> float:
        """Calculate temperature based on reviewer bias."""
        return max(0.0, min(1.0, 0.7 + (bias * 0.1)))

    def _create_previous_context(self, iterations: List[Dict[str, Any]]) -> str:
        """Create context from previous reviews with proper reviewer handling."""
        if not iterations:
            return ""
            
        context = "\nPrevious reviews:\n"
        for iteration in iterations:
            if iteration.get('reviews'):
                for review in iteration['reviews']:
                    if not review.get('error', False) and review.get('reviewer') and review.get('content'):
                        if review.get('is_presentation', False):
                            context += f"\n{review['reviewer']} Overall Assessment:\n{review['content']}\n"
                            if review.get('slide_reviews'):
                                context += "\nKey points from slides:\n"
                                for slide_review in review['slide_reviews']:
                                    if not slide_review.get('error'):
                                        context += f"Slide {slide_review['slide_number']}: "
                                        context += f"{'; '.join(slide_review.get('key_points', []))}\n"
                        else:
                            context += f"\n{review['reviewer']}:\n{review['content']}\n"
                            
                if iteration.get('dialogue'):
                    context += "\nDiscussion Summary:\n"
                    for dialogue in iteration['dialogue']:
                        if isinstance(dialogue, dict) and dialogue.get('key_points'):
                            context += f"{dialogue['key_points']}\n"
        
        return context

    def process_review(self, content: str, sections: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Process document review with iterative discussions and comprehensive final analysis."""
        iterations = []
        
        try:
            # Validate config and reviewer information
            if not config or not isinstance(config, dict):
                raise ValueError("Invalid configuration")
                
            if not config.get('reviewers') or not isinstance(config['reviewers'], list):
                raise ValueError("No reviewers specified in configuration")
                
            reviewers = config['reviewers']
            
            # Define is_presentation at the start
            is_presentation = any(section.get('type') == 'slide' for section in sections)
            if is_presentation:
                config['doc_type'] = "Presentation"
            
            temperature = self._calculate_temperature(config.get('bias', 0))
            model = self.model_config.get(config['doc_type'], "gpt-4o")
            
            for iteration in range(config['iterations']):
                logging.info(f"Starting iteration {iteration + 1}")
                iteration_reviews = []
                
                # Generate reviews using context from all previous iterations
                for rev_idx, reviewer in enumerate(reviewers):
                    try:
                        logging.info(f"Processing reviewer {rev_idx + 1}: {reviewer['expertise']}")
                        
                        # Create context from all previous iterations for this reviewer
                        previous_context = self._create_review_context(iterations, reviewer['expertise'])
                        
                        if is_presentation:
                            presentation_reviewer = PresentationReviewer(model=model)
                            review_results = presentation_reviewer.review_presentation(
                                sections=sections,
                                expertise=reviewer['expertise'],
                                temperature=temperature
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
                        
                        # Ensure reviewer information is present
                        if not review.get('reviewer'):
                            review['reviewer'] = reviewer['expertise']
                        
                        iteration_reviews.append(review)
                        logging.info(f"Generated review for {review['reviewer']}")
                        
                    except Exception as e:
                        error_msg = f"Error in review by {reviewer['expertise']}: {str(e)}"
                        logging.error(error_msg)
                        iteration_reviews.append({
                            'reviewer': reviewer['expertise'],
                            'content': error_msg,
                            'timestamp': datetime.now().isoformat(),
                            'error': True
                        })
                
                # Create the current iteration data
                current_iteration = {
                    'iteration_number': iteration + 1,
                    'reviews': iteration_reviews,
                    'reviewers': [r['expertise'] for r in reviewers]
                }
                
                # Generate reviewer dialogue for non-presentation reviews
                if not is_presentation and iteration_reviews:
                    try:
                        valid_reviews = [r for r in iteration_reviews if not r.get('error', False)]
                        logging.info(f"Found {len(valid_reviews)} valid reviews for dialogue")
                        
                        if valid_reviews:
                            dialogue = self._generate_reviewer_dialogue(
                                reviews=valid_reviews,
                                previous_iterations=iterations,  # Pass all previous iterations
                                config=config
                            )
                            
                            if dialogue and not dialogue.get('error'):
                                current_iteration['dialogue'] = [dialogue]
                            else:
                                current_iteration['dialogue'] = []
                        else:
                            current_iteration['dialogue'] = []
                            
                    except Exception as e:
                        logging.error(f"Error generating dialogue: {str(e)}")
                        current_iteration['dialogue'] = []
                
                iterations.append(current_iteration)
                
            # Generate final analysis considering all iterations
            try:
                logging.info("Generating final analysis")
                moderator_summary = self.moderator.summarize_discussion(iterations)
            except Exception as e:
                logging.error(f"Error generating moderator summary: {str(e)}")
                moderator_summary = f"Error generating final analysis: {str(e)}"
            
            return {
                'iterations': iterations,
                'moderator_summary': moderator_summary,
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'doc_type': config['doc_type'],
                'is_presentation': is_presentation,
                'reviewers': [r['expertise'] for r in reviewers]
            }
            
        except Exception as e:
            logging.error(f"Error in review process: {str(e)}")
            raise

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
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Select Expertise Areas")
        for i in range(num_reviewers):
            st.markdown(f"#### Reviewer {i+1}")
            expertise_category = st.selectbox(
                "Expertise Category",
                list(EXPERTISE_OPTIONS.keys()),
                key=f"cat_{i}"
            )
            expertise = st.selectbox(
                "Specific Expertise",
                EXPERTISE_OPTIONS[expertise_category],
                key=f"exp_{i}"
            )
            reviewers.append({"expertise": expertise})
    
    with col2:
        if doc_type == "Grant" and scoring_system == "nih":
            st.markdown("### NIH Review Criteria")
            st.info("""
            Reviews will follow NIH criteria:
            
            1. **Significance**
               - Important problem/critical barrier assessment
               - Scientific/technical/clinical advancement potential
               - Impact on concepts/methods/technologies
            
            2. **Innovation**
               - Challenge to research paradigms
               - Novel concepts/approaches/methods
               - State-of-the-art advancement
            
            3. **Approach**
               - Strategy and methodology assessment
               - Risk management and alternatives
               - Timeline and resource feasibility
            """)
    
    # Document upload
    st.markdown("## Document Upload")
    file_type_help = """
    Supported formats:
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
