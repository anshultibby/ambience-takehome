"""
UI Components and Styling for the ICD-10 Code Assistant
Separated from main app for better organization and readability
"""

import streamlit as st
import pandas as pd
import json
from typing import List, Dict, Any


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 100%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Fix Streamlit rerun buttons */
    .stAlert > div {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    /* Contain log content with better readability */
    .log-container {
        max-height: 500px;
        overflow-y: auto;
        border: 2px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Source Code Pro', monospace;
    }
    
    /* Style step headers */
    .step-header {
        background-color: #f0f8ff;
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
        font-weight: bold;
        color: #2c3e50;
    }
    
    /* Style AI responses */
    .ai-response {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
        line-height: 1.6;
        color: #333;
    }
    
    /* Style tool calls */
    .tool-call {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    
    /* Style tool results */
    .tool-result {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
    
    /* Prevent text overflow */
    .stMarkdown, .stText, .stWrite {
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
    }
    
    /* Fix JSON display */
    .stJson {
        max-width: 100%;
        overflow-x: auto;
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 6px !important;
        padding: 12px !important;
    }
    
    /* Improve separator lines */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, #f0f0f0, #ccc, #f0f0f0);
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar_config(env_api_key: str = None, env_gemini_key: str = None) -> tuple[str, int]:
    """Render the sidebar configuration and return api_key and max_iterations"""
    with st.sidebar:
        st.header("Configuration")
        
        # API Key handling - prioritize Gemini
        gemini_key = None
        if env_gemini_key:
            st.success("🔑 Gemini API Key loaded from environment")
            gemini_key = env_gemini_key
            if st.checkbox("Override with custom Gemini API key"):
                gemini_key = st.text_input(
                    "Custom Gemini API Key", 
                    type="password",
                    help="Override the environment Gemini API key"
                )
        else:
            gemini_key = st.text_input(
                "Gemini API Key", 
                type="password",
                help="Enter your Google Gemini API key, or set GEMINI_API_KEY environment variable"
            )
        
        # Fallback to OpenAI if no Gemini key provided
        if not gemini_key:
            st.info("💡 No Gemini API key provided. You can also use OpenAI API key as fallback.")
            if env_api_key:
                st.success("🔑 OpenAI API Key loaded from environment (fallback)")
                api_key = env_api_key
                if st.checkbox("Override with custom OpenAI API key"):
                    api_key = st.text_input(
                        "Custom OpenAI API Key", 
                        type="password",
                        help="Override the environment OpenAI API key"
                    )
            else:
                api_key = st.text_input(
                    "OpenAI API Key (fallback)", 
                    type="password",
                    help="Enter your OpenAI API key as fallback, or set OPENAI_API_KEY environment variable"
                )
        else:
            api_key = gemini_key
            
        # Show instructions for setting up .env file
        with st.expander("💡 How to use .env file", expanded=False):
            st.markdown("""
            **For better security, use a .env file:**
            
            1. Copy `env.example` to `.env`
            2. Edit `.env` and add your API keys:
               ```
               GEMINI_API_KEY=your_gemini_api_key_here
               OPENAI_API_KEY=your_openai_api_key_here
               ```
            3. Restart the application
            
            **Priority:** Gemini API key is preferred, OpenAI is used as fallback.
            The `.env` file is ignored by git for security.
            """)
        
        # File status check
        render_file_status()
        
        # Max iterations slider
        max_iterations = st.slider(
            "Max Processing Steps", 
            min_value=5, 
            max_value=50, 
            value=20,
            help="Maximum number of AI processing steps"
        )
    
    return api_key, max_iterations


def render_file_status():
    """Check and display the status of required XML files"""
    import os
    
    st.header("File Status")
    
    alphabetical_path = "icd10cm_index_2025.xml"
    tabular_path = "icd10cm_tabular_2025.xml"
    
    if os.path.exists(alphabetical_path):
        st.success("✅ Alphabetical Index XML found")
    else:
        st.error("❌ Alphabetical Index XML not found")
        
    if os.path.exists(tabular_path):
        st.success("✅ Tabular List XML found")
    else:
        st.error("❌ Tabular List XML not found")


def render_input_section() -> str:
    """Render the transcript input section and return the transcript"""
    st.header("Input Transcript")
    
    # Sample transcript for testing
    sample_transcript = """Patient is a 65-year-old male with a history of type 2 diabetes mellitus. 
He presents today with complaints of severe leg pain and swelling in both lower extremities. 
His A1c is 12, indicating poor glycemic control. He is currently on insulin therapy.
Patient also reports high triglycerides and has been diagnosed with hypertension.
BMI is 39, indicating class 2 obesity. There are concerns about diabetic neuropathy 
and potential peripheral artery disease affecting blood supply to the legs.
Patient has poor wound healing and kidney function needs monitoring."""
    
    if st.button("Load Sample Transcript"):
        st.session_state['transcript'] = sample_transcript
    
    transcript = st.text_area(
        "Medical Transcript",
        value=st.session_state.get('transcript', ''),
        height=300,
        placeholder="Paste the medical transcript here..."
    )
    
    return transcript


def render_summary_metrics(entries: List[Dict[str, Any]]):
    """Render summary metrics for the extracted entries"""
    total_entries = len(entries)
    confident_entries = sum(1 for entry in entries if entry.get('confidence') == 'confident')
    review_entries = sum(1 for entry in entries if entry.get('confidence') == 'requires_human_review')
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    with col_metrics1:
        st.metric("Total Codes", total_entries)
    with col_metrics2:
        confidence_pct = f"{confident_entries/total_entries*100:.0f}%" if total_entries > 0 else "0%"
        st.metric("Confident", confident_entries, delta=confidence_pct)
    with col_metrics3:
        review_pct = f"{review_entries/total_entries*100:.0f}%" if total_entries > 0 else "0%"
        st.metric("Need Review", review_entries, delta=review_pct)


def render_entry_details(entry: Dict[str, Any], confidence_type: str):
    """Render the details of a single ICD-10 entry"""
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.write("**Primary Code:**", entry.get('icd10_code', 'N/A'))
        st.write("**Condition:**", entry.get('icd10_condition_name', 'N/A'))
        st.write("**Reasoning:**", entry.get('reasoning', 'N/A'))
        
        if 'alternate_codes' in entry and entry['alternate_codes']:
            st.write("**Alternate Codes:**")
            for code in entry['alternate_codes']:
                st.write(f"- {code}")
    
    with col_right:
        if confidence_type == 'confident':
            st.success("**Confidence: High**")
            if 'confidence_reasoning' in entry:
                st.write("**Why confident:**")
                st.write(entry['confidence_reasoning'])
        elif confidence_type == 'review':
            st.warning("**Confidence: Requires Review**")
            if 'confidence_reasoning' in entry:
                st.write("**Review needed because:**")
                st.write(entry['confidence_reasoning'])


def render_entries_by_confidence(entries: List[Dict[str, Any]]):
    """Render entries grouped by confidence level"""
    # Group entries by confidence level
    confident_list = [entry for entry in entries if entry.get('confidence') == 'confident']
    review_list = [entry for entry in entries if entry.get('confidence') == 'requires_human_review']
    no_confidence_list = [entry for entry in entries if 'confidence' not in entry]
    
    # Display confident entries first
    if confident_list:
        st.success(f"✅ **Confident Codes ({len(confident_list)})**")
        for entry in confident_list:
            code = entry.get('icd10_code', 'N/A')
            condition = entry.get('icd10_condition_name', 'N/A')
            with st.expander(f"✅ {code} - {condition}", expanded=False):
                render_entry_details(entry, 'confident')
    
    # Display entries needing review
    if review_list:
        st.warning(f"⚠️ **Codes Requiring Human Review ({len(review_list)})**")
        for entry in review_list:
            code = entry.get('icd10_code', 'N/A')
            condition = entry.get('icd10_condition_name', 'N/A')
            with st.expander(f"⚠️ {code} - {condition}", expanded=False):
                render_entry_details(entry, 'review')
    
    # Display entries without confidence scores (fallback)
    if no_confidence_list:
        st.info(f"ℹ️ **Other Codes ({len(no_confidence_list)})**")
        for i, entry in enumerate(no_confidence_list, 1):
            code = entry.get('icd10_code', 'N/A')
            with st.expander(f"Entry {i}: {code}", expanded=True):
                render_entry_details(entry, 'other')


def render_export_buttons(entries: List[Dict[str, Any]]):
    """Render export functionality buttons"""
    st.divider()
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("📥 Export Results as JSON"):
            results_json = json.dumps(entries, indent=2)
            st.download_button(
                label="Download JSON",
                data=results_json,
                file_name="icd10_analysis_results.json",
                mime="application/json"
            )
    
    with col_export2:
        if st.button("📊 Export Summary as CSV"):
            summary_data = []
            for i, entry in enumerate(entries, 1):
                reasoning = entry.get('reasoning', 'N/A')
                truncated_reasoning = reasoning[:100] + '...' if len(reasoning) > 100 else reasoning
                
                summary_data.append({
                    'Entry': i,
                    'ICD10_Code': entry.get('icd10_code', 'N/A'),
                    'Condition_Name': entry.get('icd10_condition_name', 'N/A'),
                    'Confidence': entry.get('confidence', 'Unknown'),
                    'Reasoning': truncated_reasoning
                })
            
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="icd10_summary.csv",
                mime="text/csv"
            )


def render_processing_details(conversation_log: List[Dict[str, Any]]):
    """Render the processing details and logs section"""
    st.header("📋 Processing Details")
    
    # Show pipeline summary
    with st.expander("Pipeline Summary", expanded=False):
        st.info("""
        **Enhanced 3-Step Pipeline:**
        1. **Preprocessing** - Extract medical information, remove conversational fluff
        2. **ICD-10 Extraction** - AI analyzes cleaned transcript and looks up codes
        3. **Confidence Scoring** - Quality assessment of extracted codes
        """)
    
    # Show detailed conversation log if available
    with st.expander("View detailed AI conversation log", expanded=False):
        if conversation_log:
            for i, log_entry in enumerate(conversation_log):
                if isinstance(log_entry, dict) and 'role' in log_entry:
                    # New format: chat messages
                    st.write(f"**Message {i+1} ({log_entry['role']}):**")
                    st.write(log_entry['content'])
                    st.divider()
                elif isinstance(log_entry, dict) and 'step' in log_entry:
                    # Old format: structured log entries
                    st.write(f"**Step {log_entry['step']}:**")
                    
                    if log_entry['type'] == 'assistant':
                        st.write("🤖 AI Response:")
                        st.write(log_entry['response'])
                    elif log_entry['type'] == 'tool':
                        st.write("🔧 Tool Call:")
                        st.json(log_entry['tool_call'])
                        st.write("📊 Result:")
                        st.write(log_entry['result'])
                    elif log_entry['type'] == 'error':
                        st.write("❌ Error:")
                        st.write(log_entry['message'])
                    
                    st.divider()
                else:
                    # Fallback for unexpected format
                    st.write(f"**Entry {i+1}:**")
                    st.write(str(log_entry))
                    st.divider()
        else:
            st.info("No detailed conversation log available for this session.") 