"""
ICD-10 Code Assistant - Main Streamlit Application
Refactored for better readability and maintainability
"""

import streamlit as st
import os
from typing import List, Dict, Any

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

# Import our separated modules
from ui_components import (
    apply_custom_css,
    render_sidebar_config,
    render_input_section,
    render_summary_metrics,
    render_entries_by_confidence,
    render_export_buttons,
    render_processing_details
)
from icd_processing import (
    process_transcript_full_pipeline,
    parse_entries_from_string,
    clear_old_session_state
)


def validate_requirements(api_key: str, transcript: str) -> tuple[bool, str]:
    """Validate that all requirements are met before processing"""
    if not api_key:
        return False, "Please enter your OpenAI API key in the sidebar."
    
    if not transcript.strip():
        return False, "Please enter a medical transcript."
    
    # Check for required XML files
    alphabetical_path = "icd10cm_index_2025.xml"
    tabular_path = "icd10cm_tabular_2025.xml"
    
    if not (os.path.exists(alphabetical_path) and os.path.exists(tabular_path)):
        return False, "Required XML files are missing. Please ensure icd10cm_index_2025.xml and icd10cm_tabular_2025.xml are in the current directory."
    
    return True, ""


def render_results_section():
    """Render the results section with parsed entries"""
    st.header("ICD-10 Code Results")
    
    # Display results if available
    if 'entries' in st.session_state and st.session_state['entries']:
        entries = st.session_state['entries']
        
        # Parse entries from string format if needed
        entries = parse_entries_from_string(entries)
        if not entries:
            st.error("Failed to parse entries. Please try running the analysis again.")
            return
        
        # Show summary statistics
        render_summary_metrics(entries)
        st.divider()
        
        # Display entries grouped by confidence level
        render_entries_by_confidence(entries)
        
        # Export functionality
        render_export_buttons(entries)
        
    elif 'entries' in st.session_state:
        st.info("No ICD-10 codes found. Try adjusting the transcript or check the processing log below.")
    else:
        st.info("Enter a transcript and click 'Analyze Transcript' to see results here.")


def handle_transcript_analysis(transcript: str, api_key: str, max_iterations: int):
    """Handle the transcript analysis process"""
    # Store transcript in session state
    st.session_state['transcript'] = transcript
    
    # Process the transcript with the full pipeline
    with st.spinner("Processing transcript with enhanced 3-step pipeline..."):
        try:
            st.info("🚀 Starting enhanced pipeline: Preprocessing → Extraction → Confidence Scoring")
            
            entries, conversation_log = process_transcript_full_pipeline(
                transcript, api_key, max_iterations
            )
            
            # Store results in session state
            st.session_state['entries'] = entries
            st.session_state['conversation_log'] = conversation_log
            
            # Show completion message
            if entries:
                st.success(f"✅ Analysis complete! Found {len(entries)} ICD-10 code entries.")
            else:
                st.warning("⚠️ No ICD-10 codes were extracted. Please check the transcript and try again.")
                
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.exception(e)  # Show full traceback for debugging


def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="ICD-10 Code Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    # Clear old session state format
    clear_old_session_state()
    
    # Apply custom styling
    apply_custom_css()
    
    # Page header
    st.title("🏥 ICD-10 Code Assistant")
    st.markdown("Enter a medical transcript to get ICD-10 code suggestions using AI-powered analysis.")
    
    # Render sidebar configuration
    env_api_key = os.getenv('OPENAI_API_KEY')
    env_gemini_key = os.getenv('GEMINI_API_KEY')
    api_key, max_iterations = render_sidebar_config(env_api_key, env_gemini_key)
    
    # Main interface with two columns
    col1, col2 = st.columns([1, 1])
    
    # Left column: Input section
    with col1:
        transcript = render_input_section()
        
        # Process button
        if st.button("🔍 Analyze Transcript", type="primary"):
            # Validate requirements
            is_valid, error_message = validate_requirements(api_key, transcript)
            
            if not is_valid:
                st.error(error_message)
            else:
                handle_transcript_analysis(transcript, api_key, max_iterations)
    
    # Right column: Results section
    with col2:
        render_results_section()
    
    # Processing details section (full width)
    if 'conversation_log' in st.session_state:
        render_processing_details(st.session_state['conversation_log'])


if __name__ == "__main__":
    main() 