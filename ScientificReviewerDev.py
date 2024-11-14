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
    tab_titles = [f"Iteration {i+1}" for i in range(len(results['iterations']))]
    tabs = st.tabs(tab_titles)

    for i, tab in enumerate(tabs):
        with tab:
            display_iteration_results(results['iterations'][i])

    # Moderator analysis
    if results.get('moderation'):
        with st.expander("Moderator Analysis", expanded=True):
            display_moderation_results(results['moderation'])
