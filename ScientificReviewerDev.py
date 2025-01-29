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

logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)

def create_review_agents(num_agents: int, review_type: str = "paper", include_moderator: bool = False) -> List[ChatOpenAI]:
    """Create review agents including a moderator if specified."""
    # Select model based on review type
    model = "gpt-4o"
    
    # Create regular review agents
    agents = [ChatOpenAI(temperature=0.1, openai_api_key=api_key, model=model) 
             for _ in range(num_agents)]
    
    # Add moderator agent if requested and multiple reviewers
    if include_moderator and num_agents > 1:
        moderator_agent = ChatOpenAI(temperature=0.1, openai_api_key=api_key, 
                                   model="gpt-4o")
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
    """Generate a debate-style prompt for reviewers to respond to previous reviews."""
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

Focus on building a constructive dialogue and improving the quality of the review.
"""
    
    return prompt

def process_chunks_with_debate(chunks: List[str], agent: ChatOpenAI, expertise: str, 
                             prompt: str, iteration: int) -> str:
    """Process multiple chunks of content for a single review iteration."""
    chunk_reviews = []
    
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"""Reviewing part {i+1} of {len(chunks)}:

{prompt}

Content part {i+1}/{len(chunks)}:
{chunk}"""

        try:
            response = agent.invoke([HumanMessage(content=chunk_prompt)])
            chunk_review = extract_content(response, f"[Error processing chunk {i+1}]")
            chunk_reviews.append(chunk_review)
        except Exception as e:
            logging.error(f"Error processing chunk {i+1} for {expertise}: {str(e)}")
            chunk_reviews.append(f"[Error in chunk {i+1}]")
    
    if len(chunks) > 1:
        compilation_prompt = f"""Please compile your reviews of all {len(chunks)} parts into a single coherent review.

Previous chunk reviews:
{''.join(chunk_reviews)}

Please provide a consolidated review addressing all sections of the document."""

        try:
            compilation_response = agent.invoke([HumanMessage(content=compilation_prompt)])
            return extract_content(compilation_response, "[Error compiling final review]")
        except Exception as e:
            logging.error(f"Error compiling review for {expertise}: {str(e)}")
            return "\n\n".join(chunk_reviews)
    
    return chunk_reviews[0]

def process_reviews_with_debate(content: str, agents: List[ChatOpenAI], expertises: List[str], 
                              custom_prompts: List[str], review_type: str, 
                              num_iterations: int, progress_callback=None) -> Dict[str, Any]:
    """Process reviews with multiple iterations of debate between reviewers with real-time updates."""
    # Create containers for real-time display
    review_containers = {}
    iteration_containers = []
    
    # Initialize containers for each iteration
    for iteration in range(num_iterations):
        iteration_header = st.subheader(f"Iteration {iteration + 1}")
        iteration_container = st.container()
        iteration_containers.append({
            "header": iteration_header,
            "container": iteration_container
        })
        
        # Initialize containers for each reviewer in this iteration
        for expertise in expertises:
            if expertise not in review_containers:
                review_containers[expertise] = []
            with iteration_container:
                reviewer_container = st.empty()
                review_containers[expertise].append(reviewer_container)
    
    # Initialize moderator container if needed
    if len(agents) > len(expertises):
        moderator_container = st.container()
        moderator_header = moderator_container.subheader("Moderator Analysis")
        moderator_content = moderator_container.empty()
    
    # Chunk the content
    content_chunks = chunk_content(content)
    all_iterations = []
    latest_reviews = []
    
    # For each iteration
    for iteration in range(num_iterations):
        review_results = []
        
        # Update progress if callback provided
        if progress_callback:
            progress = (iteration / num_iterations) * 100
            progress_callback(progress, f"Processing iteration {iteration + 1}/{num_iterations}")
        
        # Get reviews from each agent
        for i, (agent, expertise, base_prompt) in enumerate(zip(agents[:-1], expertises, custom_prompts)):
            try:
                debate_prompt = get_debate_prompt(expertise, iteration + 1, latest_reviews, review_type)
                full_prompt = f"{base_prompt}\n\n{debate_prompt}"
                
                # Show "Generating..." placeholder
                review_containers[expertise][iteration].markdown("🔄 Generating review...")
                
                # Process chunks for this review
                review_text = process_chunks_with_debate(
                    content_chunks, agent, expertise, full_prompt, iteration + 1
                )
                
                # Update review display in real-time
                with review_containers[expertise][iteration].container():
                    st.write(f"Review by {expertise}")
                    sections = review_text.split('\n\n')
                    for section in sections:
                        st.markdown(section.strip())
                        st.markdown("---")
                
                review_result = {
                    "expertise": expertise,
                    "review": review_text,
                    "iteration": iteration + 1,
                    "success": True
                }
                
                review_results.append(review_result)
                
            except Exception as e:
                logging.error(f"Error in review process for {expertise}: {str(e)}")
                error_message = f"An error occurred while processing review from {expertise}. Error: {str(e)}"
                review_containers[expertise][iteration].error(error_message)
                
                review_results.append({
                    "expertise": expertise,
                    "review": error_message,
                    "iteration": iteration + 1,
                    "success": False
                })
        
        all_iterations.append(review_results)
        latest_reviews = review_results
    
    # After all iterations, have moderator analyze the complete discussion
    moderation_result = None
    if len(agents) > len(expertises):
        try:
            moderator_content.markdown("🔄 Generating moderator analysis...")
            
            moderator_prompt = """As a senior scientific moderator, analyze the complete review discussion:

"""
            for iteration_idx, iteration_reviews in enumerate(all_iterations, 1):
                moderator_prompt += f"\nIteration {iteration_idx}:\n"
                for review in iteration_reviews:
                    if review.get("success", False):
                        moderator_prompt += f"\nReview by {review['expertise']}:\n{review['review']}\n"
            
            moderator_prompt += """
Please provide a comprehensive analysis including:

1. DISCUSSION EVOLUTION
- How did viewpoints evolve across iterations
- Key points of convergence and divergence
- Quality and depth of the scientific discourse

2. REVIEW ANALYSIS
- Scientific rigor of each reviewer's contributions
- Strength of arguments and supporting evidence
- Constructiveness of the debate

3. SYNTHESIS OF KEY POINTS
- Areas of consensus
- Unresolved disagreements
- Most compelling arguments
- Critical insights gained through discussion

4. FINAL ASSESSMENT
- Overall score (1-9): [Score]
- Key strengths: [List 3-5 main strengths]
- Key weaknesses: [List 3-5 main weaknesses]
- Priority improvements: [List 3-5 main suggestions]
- Final recommendation: [Accept/Major Revision/Minor Revision/Reject]

Please provide specific examples from the discussion to support your analysis."""

            try:
                moderator_response = agents[-1].invoke([HumanMessage(content=moderator_prompt)])
                moderation_result = extract_content(moderator_response, "[Error: Unable to extract moderator response]")
                
                # Update moderator analysis in real-time
                with moderator_content.container():
                    sections = moderation_result.split('\n\n')
                    for section in sections:
                        st.markdown(section.strip())
                        st.markdown("---")
                
            except Exception as mod_error:
                logging.error(f"Moderator API Error: {str(mod_error)}")
                moderation_result = "Error occurred during moderation. Please try again."
                moderator_content.error(moderation_result)
            
        except Exception as e:
            logging.error(f"Error in moderation process: {str(e)}")
            moderation_result = f"An error occurred during moderation. Error: {str(e)}"
            moderator_content.error(moderation_result)
    
    return {
        "all_iterations": all_iterations,
        "moderation": moderation_result
    }

def display_review_results_with_debate(results: Dict[str, Any]) -> None:
    """Display results from iterative review process."""
    try:
        # Display iterations
        for iteration_idx, iteration_reviews in enumerate(results["all_iterations"], 1):
            st.subheader(f"Iteration {iteration_idx}")
            for review in iteration_reviews:
                with st.expander(f"Review by {review['expertise']}", expanded=True):
                    if review.get("success", False):
                        sections = review['review'].split('\n\n')
                        for section in sections:
                            st.write(section.strip())
                            st.markdown("---")
                    else:
                        st.error(review['review'])
        
        # Display final moderation
        if results["moderation"]:
            st.subheader("Final Moderator Analysis")
            if not results["moderation"].startswith("[Error"):
                sections = results["moderation"].split('\n\n')
                for section in sections:
                    st.write(section.strip())
                    st.markdown("---")
            else:
                st.error(results["moderation"])
    
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        logging.exception("Error in display_review_results_with_debate:")

def scientific_review_page():
    st.header("Multi-Agent Scientific Review System")
    
    # Add session state for storing reviewer configurations
    if 'expertises' not in st.session_state:
        st.session_state.expertises = []
    if 'custom_prompts' not in st.session_state:
        st.session_state.custom_prompts = []
    
    # Review type selection
    review_type = st.selectbox(
        "Select Review Type",
        ["Paper", "Grant", "Poster"]
    )
    
    # Number of reviewers with validation
    num_reviewers = st.number_input(
        "Number of Reviewers",
        min_value=1,
        max_value=10,
        value=2,
        key="num_reviewers"
    )
    
    # Number of iterations with validation
    num_iterations = st.number_input(
        "Number of Discussion Iterations",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of rounds of discussion between reviewers",
        key="num_iterations"
    )
    
    # Option for moderator when multiple reviewers
    use_moderator = False
    if num_reviewers > 1:
        use_moderator = st.checkbox(
            "Include Moderator/Judge Review", 
            value=True,
            key="use_moderator"
        )
    
    # Collect expertise and custom prompts for each reviewer
    expertises = []
    custom_prompts = []
    
    with st.expander("Configure Reviewers"):
        for i in range(num_reviewers):
            col1, col2 = st.columns(2)
            
            # Expertise input with unique key
            with col1:
                expertise = st.text_input(
                    f"Expertise for Reviewer {i+1}", 
                    value=f"Scientific Expert {i+1}",
                    key=f"expertise_{i}"
                )
                expertises.append(expertise)
            
            # Custom prompt input with unique key
            with col2:
                default_prompt = get_default_prompt(review_type, expertise)
                prompt = st.text_area(
                    f"Custom Prompt for Reviewer {i+1}",
                    value=default_prompt,
                    height=200,
                    key=f"prompt_{i}"
                )
                custom_prompts.append(prompt)
    
    # File upload with validation
    uploaded_file = st.file_uploader(
        f"Upload {review_type} (PDF)",
        type=["pdf"],
        key="uploaded_file"
    )
    
    # Start review button with validation
    start_review = st.button(
        "Start Review",
        disabled=not uploaded_file,  # Disable if no file uploaded
        key="start_review"
    )
    
    if uploaded_file and start_review:
        try:
            # Create a progress bar
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
            agents = create_review_agents(num_reviewers, review_type.lower(), use_moderator)
            
            # Validate inputs
            if not all(expertises) or not all(custom_prompts):
                st.error("Please ensure all reviewer configurations are complete.")
                return
            
            # Process reviews with real-time updates
            update_progress(30, "Starting review process...")
            results = process_reviews_with_debate(
                content=content,
                agents=agents,
                expertises=expertises,
                custom_prompts=custom_prompts,
                review_type=review_type.lower(),
                num_iterations=num_iterations,
                progress_callback=update_progress
            )
            
            update_progress(100, "Review process completed!")
            
            # Clear progress indicators
            time.sleep(1)  # Brief pause to show completion
            progress_bar.empty()
            status_text.empty()
            
            st.success("Review process completed successfully!")
            
        except Exception as e:
            st.error(f"An error occurred during the review process: {str(e)}")
            logging.exception("Error in review process:")
            
            if st.sidebar.checkbox("Debug Mode", value=False):
                st.exception(e)
            
            st.warning("Please try again or check your inputs.")

def get_default_prompt(review_type: str, expertise: str) -> str:
    """Get default prompt based on review type."""
    try:
        prompts = {
            "Paper": f"""As an expert in {expertise}, please review this scientific paper considering:
                
Strengths and Weaknesses

1. Scientific Merit and Novelty
2. Methodology and Technical Rigor
3. Data Analysis and Interpretation
4. Clarity and Presentation
5. Impact and Significance

Please provide scores (1-9) for each aspect and an overall score.""",
            
            "Grant": f"""As an expert in {expertise}, please evaluate this grant proposal considering:
                
Strengths and Weaknesses

1. Innovation and Significance
2. Approach and Methodology
3. Feasibility and Timeline
4. Budget Justification
5. Expected Impact

Please provide scores (1-9) for each aspect and an overall score.""",
            
            "Poster": f"""As an expert in {expertise}, please review this scientific poster considering:
                
Strengths and Weaknesses

1. Visual Appeal and Organization
2. Scientific Content
3. Methodology Presentation
4. Results and Conclusions
5. Impact and Relevance

Please provide scores (1-9) for each aspect and an overall score."""
        }
        return prompts.get(review_type, f"Please provide a thorough review of this {review_type.lower()}.")
    except Exception as e:
        logging.error(f"Error generating default prompt: {str(e)}")
        return "Please provide a thorough review of this submission."

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
            border-left: 3px solid #4CAF50;
            background-color: #f8f9fa;
        }
        .review-header {
            font-weight: bold;
            color: #2C3E50;
            margin-bottom: 0.5rem;
        }
        .iteration-header {
            background-color: #2C3E50;
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .review-score {
            font-weight: bold;
            color: #E74C3C;
        }
        .moderator-analysis {
            background-color: #ECF0F1;
            padding: 1rem;
            border-radius: 4px;
            margin-top: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Add version number and info to sidebar
    st.sidebar.text("Version 2.0.0")
    
    # Model information in sidebar
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("- Reviewer Agent Model: GPT-4o")
    st.sidebar.markdown("- Moderator Model: GPT-4o")
    
    # Additional settings in sidebar
    with st.sidebar.expander("Advanced Settings"):
        st.slider(
            "Model Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controls randomness in model responses"
        )
        st.number_input(
            "Maximum Tokens per Chunk",
            min_value=1000,
            max_value=6000,
            value=4000,
            step=500,
            help="Maximum tokens per content chunk"
        )
        st.checkbox(
            "Debug Mode",
            value=False,
            help="Show detailed logging information"
        )
    
    # Instructions/About section in sidebar
    with st.sidebar.expander("Instructions"):
        st.markdown("""
        1. Select review type (Paper/Grant/Poster)
        2. Set number of reviewers (1-10)
        3. Choose number of discussion iterations
        4. Configure reviewer expertise and prompts
        5. Upload document (PDF)
        6. Click 'Start Review' to begin
        
        The system will:
        - Process document in chunks if needed
        - Generate reviews from each expert
        - Facilitate discussion across iterations
        - Provide final moderation analysis
        """)
    
    scientific_review_page()

def error_handler(func):
    """Decorator for handling errors in functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.exception(f"Error in {func.__name__}:")
            return None
    return wrapper

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Unexpected error in main application:")
