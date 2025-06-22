import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import json
import os
from typing import List, Dict, Any
import re
import ast

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

from simple_xml_lookup import lookup_alphabetical_index, lookup_tabular_list
from descriptions import EXTRACT_ICD10_SYSTEM_PROMPT, PREPROCESSING_PROMPT, CONFIDENCE_SCORING_PROMPT
from openai_tools_converter import (
    get_openai_tools_for_icd10, 
    ICD10ContextVariables, 
    execute_function_with_context, 
    extract_tool_call_from_response
)

# Configuration
MODEL_ID = "gpt-4o"
ALPHABETICAL_INDEX_PATH = "icd10cm_index_2025.xml"
TABULAR_LIST_PATH = "icd10cm_tabular_2025.xml"

# System prompt using the template from descriptions.py
def get_system_prompt():
    tools = get_openai_tools_for_icd10()
    return EXTRACT_ICD10_SYSTEM_PROMPT.substitute(tools=tools)

def preprocess_transcript(transcript: str, api_key: str):
    """
    First step: Extract relevant medical details and remove fluff from transcript
    """
    client = OpenAI(api_key=api_key)
    
    messages = [
        {"role": "system", "content": PREPROCESSING_PROMPT},
        {"role": "user", "content": f"Please extract the relevant medical information from this transcript and remove all fluff:\n\n{transcript}"}
    ]
    
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
    )
    
    cleaned_transcript = response.choices[0].message.content
    return cleaned_transcript

def add_confidence_scores(chat_history: List[Dict], entries: List[Dict], api_key: str):
    """
    Post-processing step: Add confidence scores to each ICD-10 entry based on chat history
    """
    if not entries:
        return entries
    
    # Parse entries from string format if needed
    entries = parse_entries_from_string(entries)
    if not entries:
        return []
        
    # chat_history should already be a list of message dictionaries, no parsing needed
    if not chat_history:
        st.warning("Chat history is empty or could not be parsed. Confidence scoring may be limited.")
        chat_history = []
        
    client = OpenAI(api_key=api_key)
    
    # Format the chat history and entries for review
    chat_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    # Format entries with codes for matching
    entries_for_review = []
    for i, entry in enumerate(entries):
        entries_for_review.append({
            "entry_index": i,
            "icd10_code": entry.get('icd10_code', 'Unknown'),
            "icd10_condition_name": entry.get('icd10_condition_name', 'Unknown'),
            "reasoning": entry.get('reasoning', 'No reasoning provided')
        })
    
    messages = [
        {"role": "system", "content": CONFIDENCE_SCORING_PROMPT},
        {"role": "user", "content": f"""
Please review this ICD-10 coding conversation and assign confidence scores to each final entry.

CHAT HISTORY:
{chat_summary}

FINAL ENTRIES TO REVIEW:
{entries_for_review}

Please return ONLY the confidence assessments as a JSON list in the same order as the entries, with format:
[
    {{
        "entry_index": 0,
        "icd10_code": "E11.9",
        "confidence_reasoning": "brief explanation of why you assigned this confidence level",
        "confidence": "confident" or "requires_human_review"
    }},
    ...
]
"""}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # Get API response content
        api_response_content = response.choices[0].message.content
        
        # Check if response is empty or None
        if not api_response_content or api_response_content.strip() == "":
            raise ValueError("API returned empty response")
        
        # Try to parse JSON - handle case where response might be wrapped in markdown
        try:
            confidence_assessments = json.loads(api_response_content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', api_response_content, re.DOTALL)
            if json_match:
                confidence_assessments = json.loads(json_match.group(1))
            else:
                # Try to find JSON array directly
                json_match = re.search(r'(\[.*?\])', api_response_content, re.DOTALL)
                if json_match:
                    confidence_assessments = json.loads(json_match.group(1))
                else:
                    raise json.JSONDecodeError("No valid JSON found in response", api_response_content, 0)
        
        # Handle case where API returns a single dict instead of a list
        if isinstance(confidence_assessments, dict):
            confidence_assessments = [confidence_assessments]
        elif not isinstance(confidence_assessments, list):
            raise ValueError(f"Expected list or dict, got {type(confidence_assessments)}")
        
        # Add confidence scores to original entries using matching
        enhanced_entries = []
        for i, entry in enumerate(entries):
            enhanced_entry = entry.copy()  # Copy original entry
            
            # Find matching confidence assessment
            matching_assessment = None
            for assessment in confidence_assessments:
                try:
                    if (assessment.get('entry_index') == i or 
                        assessment.get('icd10_code') == entry.get('icd10_code')):
                        matching_assessment = assessment
                        break
                except Exception as e:
                    raise e
            
            if matching_assessment:
                enhanced_entry['confidence_reasoning'] = matching_assessment['confidence_reasoning']
                enhanced_entry['confidence'] = matching_assessment['confidence']
            else:
                # Fallback if no match found
                enhanced_entry['confidence_reasoning'] = "Unable to match confidence assessment"
                enhanced_entry['confidence'] = "requires_human_review"
            
            enhanced_entries.append(enhanced_entry)
        
        return enhanced_entries
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing failed: {e}")
        # Fallback: return original entries with default confidence
        fallback_entries = []
        for entry in entries:
            enhanced_entry = entry.copy()
            enhanced_entry['confidence_reasoning'] = "JSON parsing failed"
            enhanced_entry['confidence'] = "requires_human_review"
            fallback_entries.append(enhanced_entry)
        return fallback_entries
    except Exception as e:
        st.warning(f"Confidence scoring failed: {e}. Proceeding without confidence scores.")
        # Fallback: return original entries with default confidence
        fallback_entries = []
        for entry in entries:
            enhanced_entry = entry.copy()
            enhanced_entry['confidence_reasoning'] = "Confidence scoring failed"
            enhanced_entry['confidence'] = "requires_human_review"
            fallback_entries.append(enhanced_entry)
        return fallback_entries

def get_openai_response(transcript: str, messages: List[Dict], api_key: str):
    """Get response from OpenAI API"""
    client = OpenAI(api_key=api_key)
    
    if len(messages) == 0:
        system_prompt = get_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the transcript: {transcript}\n\nWhat's your first step?"}
        ]

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
    )
    return response, messages

def extract_icd10_codes_with_chat_history(cleaned_transcript: str, api_key: str, max_iterations: int = 30, step_container=None):
    """
    Extract ICD-10 codes and return both entries and chat history for confidence scoring
    Also tracks steps for real-time display
    """
    # Initialize context
    context_vars = ICD10ContextVariables(
        xml_file_path_alphabetical=ALPHABETICAL_INDEX_PATH,
        xml_file_path_tabular=TABULAR_LIST_PATH,
        entries=[]
    )
    
    messages = []
    counter = 0
    steps_log = []  # Track steps for display
    
    # Initialize the display container once
    if step_container:
        with step_container:
            steps_header = st.empty()
            entries_status = st.empty()
            steps_display_container = st.container()
    
    while counter < max_iterations:
        try:
            # Get AI response
            response, messages = get_openai_response(cleaned_transcript, messages, api_key)
            text = response.choices[0].message.content
            
            # Log this step
            step_info = {
                'step_number': counter + 1,
                'ai_response': text,
                'tool_call': None,
                'tool_result': None,
                'status': 'processing'
            }
            
            # Check for stop condition
            if "stop" in text.lower():
                step_info['status'] = 'completed'
                step_info['ai_response'] = "🏁 Analysis complete - AI has finished processing"
                steps_log.append(step_info)
                
                # Update display if container provided
                if step_container:
                    update_steps_display(steps_log, context_vars.entries, steps_header, entries_status, steps_display_container, counter + 1)
                break
            
            # Extract and execute tool call
            tool_call = extract_tool_call_from_response(text)
            
            if tool_call:
                step_info['tool_call'] = tool_call
                
                # Execute the tool call
                result = execute_function_with_context(
                    tool_call["function_name"], 
                    tool_call["arguments"], 
                    context_vars
                )
                
                step_info['tool_result'] = result
                step_info['status'] = 'completed'
                
                # Prepare next message
                tool_call_message = {
                    "role": "user", 
                    "content": f"Function returned: {result}. What's your next step?"
                }
            else:
                step_info['status'] = 'no_tool_call'
                tool_call_message = {
                    "role": "user", 
                    "content": "No function call found. What's your next step?"
                }

            # Add step to log
            steps_log.append(step_info)
            
            # Update display if container provided - only add the new step
            if step_container:
                update_steps_display(steps_log, context_vars.entries, steps_header, entries_status, steps_display_container, counter + 1)

            # Update messages
            messages += [
                {"role": "assistant", "content": text},
                tool_call_message
            ]
            
            counter += 1
            
        except Exception as e:
            # Log error step
            error_step = {
                'step_number': counter + 1,
                'ai_response': f"Error occurred: {str(e)}",
                'tool_call': None,
                'tool_result': None,
                'status': 'error'
            }
            steps_log.append(error_step)
            
            # Update display if container provided
            if step_container:
                update_steps_display(steps_log, context_vars.entries, steps_header, entries_status, steps_display_container, counter + 1)
            
            st.error(f"Error in extraction step {counter + 1}: {str(e)}")
            break
    
    return context_vars.entries, messages

def update_steps_display(steps_log, current_entries, steps_header, entries_status, steps_container, current_step_num):
    """Update the steps display by only updating headers and adding new steps"""
    if not steps_log:
        return
    
    # Update headers
    steps_header.write(f"**AI Processing Steps:** {len(steps_log)} steps completed")
    
    if current_entries:
        entries_status.success(f"📊 **Entries found so far:** {len(current_entries)}")
    else:
        entries_status.empty()
    
    # Only display the latest step to avoid duplicates
    latest_step = steps_log[-1]
    step_num = latest_step['step_number']
    status = latest_step['status']
    
    # Determine status icon and color
    if status == 'completed':
        status_icon = "✅"
        status_text = "Completed"
    elif status == 'processing':
        status_icon = "🔄"
        status_text = "Processing"
    elif status == 'error':
        status_icon = "❌"
        status_text = "Error"
    elif status == 'no_tool_call':
        status_icon = "💭"
        status_text = "Thinking"
    else:
        status_icon = "ℹ️"
        status_text = "Unknown"
    
    # Create expander title - remove step numbers, focus on action
    if latest_step.get('tool_call'):
        tool_name = latest_step['tool_call'].get('function_name', 'unknown')
        tool_args = latest_step['tool_call'].get('arguments', {})
        
        # Make tool names more readable and include the key parameter
        if tool_name == 'lookup_alphabetical_index':
            query = tool_args.get('query', 'unknown')
            action_name = f"Search Alphabetical Index: '{query}'"
        elif tool_name == 'lookup_tabular_list':
            code = tool_args.get('code', 'unknown')
            action_name = f"Search Tabular List: '{code}'"
        elif tool_name == 'add_entry' or tool_name == 'create_icd10_suspect_entry':
            # For add_entry, show the ICD code being added
            if isinstance(tool_args, dict):
                if 'icd10_suspect_entry' in tool_args:
                    entry_data = tool_args['icd10_suspect_entry']
                    if isinstance(entry_data, dict):
                        code = entry_data.get('icd10_code', 'unknown')
                    else:
                        code = 'unknown'
                else:
                    code = tool_args.get('icd10_code', 'unknown')
                action_name = f"Add ICD-10 Entry: '{code}'"
            else:
                action_name = "Add ICD-10 Entry"
        else:
            action_name = tool_name.replace('_', ' ').title()
        
        expander_title = f"{status_icon} {action_name} - {status_text}"
    else:
        expander_title = f"{status_icon} AI Analysis - {status_text}"
    
    # Add the new step to the container
    with steps_container:
        # Create expander (expand first few steps by default)
        expanded = step_num <= 2
        
        with st.expander(expander_title, expanded=expanded):
            # Show AI response
            st.write("**🤖 AI Response:**")
            ai_response = latest_step['ai_response']
            if len(ai_response) > 500:
                st.write(ai_response[:500] + "...")
                with st.expander("Show full AI response", expanded=False):
                    st.write(ai_response)
            else:
                st.write(ai_response)
            
            # Show tool call if present
            if latest_step.get('tool_call'):
                st.write("**🔧 Action Details:**")
                tool_call = latest_step['tool_call']
                st.write(f"Function: `{tool_call.get('function_name', 'unknown')}`")
                
                if tool_call.get('arguments'):
                    with st.expander("View parameters", expanded=False):
                        st.json(tool_call['arguments'])
                
                # Show tool result
                if latest_step.get('tool_result'):
                    st.write("**📊 Result:**")
                    result = latest_step['tool_result']
                    if isinstance(result, str) and len(result) > 300:
                        st.write(result[:300] + "...")
                        with st.expander("Show full result", expanded=False):
                            st.write(result)
                    else:
                        st.write(result)

def process_transcript_full_pipeline(transcript: str, api_key: str, max_iterations: int = 30):
    """
    Complete pipeline: preprocessing → extraction → confidence scoring
    """
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Stage 1: Preprocess transcript
    status_text.text("Stage 1/3: Preprocessing transcript...")
    progress_bar.progress(0.1)
    
    with st.expander("🔄 Stage 1: Preprocessing", expanded=False):
        st.info("Extracting relevant medical information and removing conversational fluff...")
        try:
            cleaned_transcript = preprocess_transcript(transcript, api_key)
            st.success("✅ Preprocessing complete")
            
            # Show cleaned transcript in a collapsible section
            with st.expander("View Cleaned Transcript", expanded=False):
                st.write(cleaned_transcript)
                
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {e}")
            return [], []
    
    progress_bar.progress(0.3)
    
    # Stage 2: Extract ICD-10 codes with live step tracking
    status_text.text("Stage 2/3: Extracting ICD-10 codes...")
    
    with st.expander("🔄 Stage 2: ICD-10 Code Extraction", expanded=True):
        st.info("AI is analyzing the cleaned transcript and looking up ICD-10 codes...")
        st.write("---")
        
        # Create a persistent container for live extraction steps
        extraction_steps_container = st.container()
        
        try:
            entries, chat_history = extract_icd10_codes_with_chat_history(
                cleaned_transcript, api_key, max_iterations, extraction_steps_container
            )
            
            st.write("---")
            if entries:
                st.success(f"✅ **Stage 2 Complete:** Extracted {len(entries)} ICD-10 code entries")
                
                # Show preliminary results
                with st.expander("View Preliminary Results (Before Confidence Scoring)", expanded=False):
                    for i, entry in enumerate(entries, 1):
                        st.write(f"**Entry {i}:** {entry.get('icd10_code', 'N/A')} - {entry.get('icd10_condition_name', 'N/A')}")
            else:
                st.warning("⚠️ **Stage 2 Complete:** No ICD-10 codes were extracted")
                
        except Exception as e:
            st.error(f"❌ Stage 2 failed: Code extraction error: {e}")
            return [], []
    
    progress_bar.progress(0.7)
    
    # Stage 3: Add confidence scores
    status_text.text("Stage 3/3: Adding confidence scores...")
    
    with st.expander("🔄 Stage 3: Confidence Scoring", expanded=False):
        st.info("Analyzing extraction quality and assigning confidence scores...")
        
        try:
            if entries:
                scored_entries = add_confidence_scores(chat_history, entries, api_key)
                
                # Show confidence summary
                confident_count = sum(1 for entry in scored_entries if entry.get('confidence') == 'confident')
                review_count = sum(1 for entry in scored_entries if entry.get('confidence') == 'requires_human_review')
                
                st.success(f"✅ **Stage 3 Complete:** Confidence scoring finished")
                st.info(f"📊 **Confidence Summary:** {confident_count} confident, {review_count} require human review")
                
            else:
                scored_entries = []
                st.info("ℹ️ **Stage 3 Complete:** No entries to score")
                
        except Exception as e:
            st.error(f"❌ Stage 3 failed: Confidence scoring error: {e}")
            scored_entries = entries  # Fall back to entries without confidence scores
    
    progress_bar.progress(1.0)
    status_text.text("✅ All stages complete!")
    
    return scored_entries, chat_history

def main():
    st.set_page_config(
        page_title="ICD-10 Code Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    # Clear old conversation log format if it exists
    if 'conversation_log' in st.session_state:
        try:
            # Check if the log has the old format and clear it
            if (st.session_state['conversation_log'] and 
                isinstance(st.session_state['conversation_log'], list) and 
                len(st.session_state['conversation_log']) > 0 and
                isinstance(st.session_state['conversation_log'][0], dict) and
                'step' in st.session_state['conversation_log'][0]):
                # This is the old format, clear it
                del st.session_state['conversation_log']
        except:
            # If there's any error checking the format, just clear it to be safe
            if 'conversation_log' in st.session_state:
                del st.session_state['conversation_log']
    
    # Add CSS to prevent overflow and contain elements
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
    
    st.title("🏥 ICD-10 Code Assistant")
    st.markdown("Enter a medical transcript to get ICD-10 code suggestions using AI-powered analysis.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Check if API key is available from environment
        env_api_key = os.getenv('OPENAI_API_KEY')
        
        if env_api_key:
            st.success("🔑 API Key loaded from environment")
            api_key = env_api_key
            # Show option to override
            if st.checkbox("Override with custom API key"):
                api_key = st.text_input(
                    "Custom OpenAI API Key", 
                    type="password",
                    help="Override the environment API key"
                )
        else:
            # API Key input
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key, or set OPENAI_API_KEY environment variable"
            )
            
            # Show instructions for setting up .env file
            with st.expander("💡 How to use .env file", expanded=False):
                st.markdown("""
                **For better security, use a .env file:**
                
                1. Copy `env.example` to `.env`
                2. Edit `.env` and add your API key:
                   ```
                   OPENAI_API_KEY=your_actual_api_key_here
                   ```
                3. Restart the application
                
                The `.env` file is ignored by git for security.
                """)
        
        # File path checks
        st.header("File Status")
        if os.path.exists(ALPHABETICAL_INDEX_PATH):
            st.success("✅ Alphabetical Index XML found")
        else:
            st.error("❌ Alphabetical Index XML not found")
            
        if os.path.exists(TABULAR_LIST_PATH):
            st.success("✅ Tabular List XML found")
        else:
            st.error("❌ Tabular List XML not found")
        
        # Max iterations
        max_iterations = st.slider(
            "Max Processing Steps", 
            min_value=5, 
            max_value=50, 
            value=20,
            help="Maximum number of AI processing steps"
        )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
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
        
        # Process button
        if st.button("🔍 Analyze Transcript", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif not transcript.strip():
                st.error("Please enter a medical transcript.")
            elif not (os.path.exists(ALPHABETICAL_INDEX_PATH) and os.path.exists(TABULAR_LIST_PATH)):
                st.error("Required XML files are missing. Please ensure icd10cm_index_2025.xml and icd10cm_tabular_2025.xml are in the current directory.")
            else:
                # Store transcript in session state
                st.session_state['transcript'] = transcript
                
                # Process the transcript with the full pipeline
                with st.spinner("Processing transcript with enhanced 3-step pipeline..."):
                    try:
                        st.info("🚀 Starting enhanced pipeline: Preprocessing → Extraction → Confidence Scoring")
                        
                        entries, conversation_log = process_transcript_full_pipeline(transcript, api_key, max_iterations)
                        st.session_state['entries'] = entries
                        st.session_state['conversation_log'] = conversation_log
                        
                        if entries:
                            st.success(f"✅ Analysis complete! Found {len(entries)} ICD-10 code entries.")
                        else:
                            st.warning("⚠️ No ICD-10 codes were extracted. Please check the transcript and try again.")
                            
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.exception(e)  # Show full traceback for debugging
    
    with col2:
        st.header("ICD-10 Code Results")
        
        # Display results if available
        if 'entries' in st.session_state and st.session_state['entries']:
            entries = st.session_state['entries']
            
            # Parse entries from string format if needed (e.g., when loaded from CSV)
            entries = parse_entries_from_string(entries)
            if not entries:
                st.error("Failed to parse entries. Please try running the analysis again.")
                return
            
            # Show summary statistics
            total_entries = len(entries)
            confident_entries = sum(1 for entry in entries if entry.get('confidence') == 'confident')
            review_entries = sum(1 for entry in entries if entry.get('confidence') == 'requires_human_review')
            
            # Summary metrics
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            with col_metrics1:
                st.metric("Total Codes", total_entries)
            with col_metrics2:
                st.metric("Confident", confident_entries, delta=f"{confident_entries/total_entries*100:.0f}%" if total_entries > 0 else "0%")
            with col_metrics3:
                st.metric("Need Review", review_entries, delta=f"{review_entries/total_entries*100:.0f}%" if total_entries > 0 else "0%")
            
            st.divider()
            
            # Group entries by confidence level
            confident_list = [entry for entry in entries if entry.get('confidence') == 'confident']
            review_list = [entry for entry in entries if entry.get('confidence') == 'requires_human_review']
            no_confidence_list = [entry for entry in entries if 'confidence' not in entry]
            
            # Display confident entries first
            if confident_list:
                st.success(f"✅ **Confident Codes ({len(confident_list)})**")
                for i, entry in enumerate(confident_list, 1):
                    with st.expander(f"✅ {entry.get('icd10_code', 'N/A')} - {entry.get('icd10_condition_name', 'N/A')}", expanded=False):
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
                            st.success("**Confidence: High**")
                            if 'confidence_reasoning' in entry:
                                st.write("**Why confident:**")
                                st.write(entry['confidence_reasoning'])
            
            # Display entries needing review
            if review_list:
                st.warning(f"⚠️ **Codes Requiring Human Review ({len(review_list)})**")
                for i, entry in enumerate(review_list, 1):
                    with st.expander(f"⚠️ {entry.get('icd10_code', 'N/A')} - {entry.get('icd10_condition_name', 'N/A')}", expanded=False):
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
                            st.warning("**Confidence: Requires Review**")
                            if 'confidence_reasoning' in entry:
                                st.write("**Review needed because:**")
                                st.write(entry['confidence_reasoning'])
            
            # Display entries without confidence scores (fallback)
            if no_confidence_list:
                st.info(f"ℹ️ **Other Codes ({len(no_confidence_list)})**")
                for i, entry in enumerate(no_confidence_list, 1):
                    with st.expander(f"Entry {i}: {entry.get('icd10_code', 'N/A')}", expanded=True):
                        st.write("**Primary Code:**", entry.get('icd10_code', 'N/A'))
                        st.write("**Condition:**", entry.get('icd10_condition_name', 'N/A'))
                        st.write("**Reasoning:**", entry.get('reasoning', 'N/A'))
                        
                        if 'alternate_codes' in entry and entry['alternate_codes']:
                            st.write("**Alternate Codes:**")
                            for code in entry['alternate_codes']:
                                st.write(f"- {code}")
            
            # Export results
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
                    # Create a summary DataFrame
                    summary_data = []
                    for i, entry in enumerate(entries, 1):
                        summary_data.append({
                            'Entry': i,
                            'ICD10_Code': entry.get('icd10_code', 'N/A'),
                            'Condition_Name': entry.get('icd10_condition_name', 'N/A'),
                            'Confidence': entry.get('confidence', 'Unknown'),
                            'Reasoning': entry.get('reasoning', 'N/A')[:100] + '...' if len(entry.get('reasoning', '')) > 100 else entry.get('reasoning', 'N/A')
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="icd10_summary.csv",
                        mime="text/csv"
                    )
        
        elif 'entries' in st.session_state:
            st.info("No ICD-10 codes found. Try adjusting the transcript or check the processing log below.")
        else:
            st.info("Enter a transcript and click 'Analyze Transcript' to see results here.")
    
    # Processing Details and Logs
    if 'conversation_log' in st.session_state:
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
            if st.session_state['conversation_log']:
                # Handle both old format (with 'step' key) and new format (chat messages)
                for i, log_entry in enumerate(st.session_state['conversation_log']):
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

def parse_entries_from_string(entries):
    """
    Helper function to parse entries that might be stored as strings
    """
    if not entries:
        return []
    
    if entries is None:
        return []
    
    if isinstance(entries, str):
        try:
            parsed = ast.literal_eval(entries)
            return parsed if parsed else []
        except:
            # If parsing fails, return empty list instead of error
            return []
    
    if isinstance(entries, list):
        parsed_entries = []
        for entry in entries:
            if isinstance(entry, str):
                try:
                    parsed_entry = ast.literal_eval(entry)
                    parsed_entries.append(parsed_entry)
                except:
                    # Skip entries that can't be parsed instead of failing completely
                    continue
            elif isinstance(entry, dict):
                parsed_entries.append(entry)
            else:
                # Skip unexpected entry types instead of failing
                continue
        return parsed_entries
    
    return entries

if __name__ == "__main__":
    main() 