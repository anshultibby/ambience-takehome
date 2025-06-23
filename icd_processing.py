"""
Core ICD-10 Processing Logic
Separated from UI for better organization and testability
"""

import streamlit as st
from typing import List, Dict, Any, Tuple
import ast

# Import the working functions from pipeline_functions
from pipeline_functions import (
    preprocess_transcript, 
    add_confidence_scores,
    get_icd10_codes_with_full_pipeline,
    get_system_prompt,
    get_response
)
from openai_tools_converter import (
    ICD10ContextVariables, 
    execute_function_with_context, 
    extract_tool_call_from_response
)

# Configuration
MODEL_ID = "gemini-2.5-flash"
ALPHABETICAL_INDEX_PATH = "icd10cm_index_2025.xml"
TABULAR_LIST_PATH = "icd10cm_tabular_2025.xml"


class StepTracker:
    """Helper class to track and display processing steps in Streamlit"""
    
    def __init__(self, step_container=None):
        self.steps_log = []
        self.step_container = step_container
        self.steps_header = None
        self.entries_status = None
        self.steps_display_container = None
        
        if step_container:
            with step_container:
                self.steps_header = st.empty()
                self.entries_status = st.empty()
                self.steps_display_container = st.container()
    
    def add_step(self, step_info: Dict[str, Any], current_entries: List[Dict] = None):
        """Add a new step to the log"""
        self.steps_log.append(step_info)
        if self.step_container:
            self._update_display(current_entries)
    
    def _update_display(self, current_entries: List[Dict] = None):
        """Update the live display of steps"""
        if not self.steps_log:
            return
        
        # Update headers
        self.steps_header.write(f"**AI Processing Steps:** {len(self.steps_log)} steps completed")
        
        # Update entries status
        if current_entries:
            self.entries_status.success(f"📊 **Entries found so far:** {len(current_entries)}")
        else:
            self.entries_status.empty()
        
        # Display the latest step
        latest_step = self.steps_log[-1]
        self._render_step(latest_step)
    
    def _render_step(self, step_info: Dict[str, Any]):
        """Render a single step in the display"""
        status = step_info['status']
        
        # Determine status icon and text
        status_icons = {
            'completed': ("✅", "Completed"),
            'processing': ("🔄", "Processing"),
            'error': ("❌", "Error"),
            'no_tool_call': ("💭", "Thinking")
        }
        status_icon, status_text = status_icons.get(status, ("ℹ️", "Unknown"))
        
        # Create expander title
        if step_info.get('tool_call'):
            tool_name = step_info['tool_call'].get('function_name', 'unknown')
            tool_args = step_info['tool_call'].get('arguments', {})
            action_name = self._get_action_name(tool_name, tool_args)
            expander_title = f"{status_icon} {action_name} - {status_text}"
        else:
            expander_title = f"{status_icon} AI Analysis - {status_text}"
        
        # Add the step to the container
        with self.steps_display_container:
            expanded = step_info['step_number'] <= 2
            
            with st.expander(expander_title, expanded=expanded):
                # Show AI response
                st.write("**🤖 AI Response:**")
                ai_response = step_info['ai_response']
                if len(ai_response) > 500:
                    st.write(ai_response[:500] + "...")
                    with st.expander("Show full AI response", expanded=False):
                        st.write(ai_response)
                else:
                    st.write(ai_response)
                
                # Show tool call if present
                if step_info.get('tool_call'):
                    st.write("**🔧 Action Details:**")
                    tool_call = step_info['tool_call']
                    st.write(f"Function: `{tool_call.get('function_name', 'unknown')}`")
                    
                    if tool_call.get('arguments'):
                        with st.expander("View parameters", expanded=False):
                            st.json(tool_call['arguments'])
                    
                    # Show tool result
                    if step_info.get('tool_result'):
                        st.write("**📊 Result:**")
                        result = step_info['tool_result']
                        if isinstance(result, str) and len(result) > 300:
                            st.write(result[:300] + "...")
                            with st.expander("Show full result", expanded=False):
                                st.write(result)
                        else:
                            st.write(result)
    
    def _get_action_name(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Get a readable action name for the tool"""
        if tool_name == 'lookup_alphabetical_index':
            query = tool_args.get('query', 'unknown')
            return f"Search Alphabetical Index: '{query}'"
        elif tool_name == 'lookup_tabular_list':
            code = tool_args.get('code', 'unknown')
            return f"Search Tabular List: '{code}'"
        elif tool_name in ['add_entry', 'create_icd10_suspect_entry']:
            if isinstance(tool_args, dict):
                if 'icd10_suspect_entry' in tool_args:
                    entry_data = tool_args['icd10_suspect_entry']
                    if isinstance(entry_data, dict):
                        code = entry_data.get('icd10_code', 'unknown')
                    else:
                        code = 'unknown'
                else:
                    code = tool_args.get('icd10_code', 'unknown')
                return f"Add ICD-10 Entry: '{code}'"
            else:
                return "Add ICD-10 Entry"
        else:
            return tool_name.replace('_', ' ').title()


def get_icd10_codes_with_chat_history_and_steps(
    cleaned_transcript: str, 
    api_key: str, 
    max_iterations: int = 30,
    step_container=None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Streamlit-aware version of ICD-10 extraction with live step tracking
    Uses the core logic from pipeline_functions.py but adds UI integration
    """
    counter = 0
    messages = []
    context_vars = ICD10ContextVariables(
        xml_file_path_alphabetical=ALPHABETICAL_INDEX_PATH,
        xml_file_path_tabular=TABULAR_LIST_PATH,
        entries=[]
    )
    
    step_tracker = StepTracker(step_container)
    
    while counter < max_iterations:
        try:
            # Use the working function from pipeline_functions
            response, messages = get_response(
                cleaned_transcript, messages, api_key, system_prompt=get_system_prompt()
            )
            text = response.choices[0].message.content
            
            # Create step info
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
                step_tracker.add_step(step_info, context_vars.entries)
                break
            
            # Extract and execute tool call (using working functions)
            tool_call = extract_tool_call_from_response(text)
            
            if tool_call:
                step_info['tool_call'] = tool_call
                
                # Execute the tool call using working function
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

            # Add step to tracker with current entries count
            step_tracker.add_step(step_info, context_vars.entries)

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
            step_tracker.add_step(error_step, context_vars.entries)
            st.error(f"Error in extraction step {counter + 1}: {str(e)}")
            break
    
    return context_vars.entries, messages


def process_transcript_full_pipeline(
    transcript: str, 
    api_key: str, 
    max_iterations: int = 30
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Complete pipeline using the working functions from pipeline_functions module
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
            # Use the Streamlit-aware version with step tracking
            entries, chat_history = get_icd10_codes_with_chat_history_and_steps(
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
    
    # Stage 3: Add confidence scores using the working function
    status_text.text("Stage 3/3: Adding confidence scores...")
    
    with st.expander("🔄 Stage 3: Confidence Scoring", expanded=False):
        st.info("Analyzing extraction quality and assigning confidence scores...")
        
        try:
            if entries:
                # Use the working function from pipeline_functions
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


def parse_entries_from_string(entries) -> List[Dict[str, Any]]:
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


def clear_old_session_state():
    """Clear old conversation log format if it exists"""
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