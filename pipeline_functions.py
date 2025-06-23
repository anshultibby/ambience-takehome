import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from simple_xml_lookup import lookup_alphabetical_index, lookup_tabular_list
from descriptions import EXTRACT_ICD10_SYSTEM_PROMPT, PREPROCESSING_PROMPT, CONFIDENCE_SCORING_PROMPT
from openai_tools_converter import (
    get_openai_tools_for_icd10, 
    ICD10ContextVariables, 
    execute_function_with_context, 
    extract_tool_call_from_response
)

# Configuration
load_dotenv()

MODEL_ID = "gemini-2.0-flash"
ALPHABETICAL_INDEX_PATH = "data/icd10cm_index_2025.xml"
TABULAR_LIST_PATH = "data/icd10cm_tabular_2025.xml"
MAX_TURNS = 50
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

def get_client_and_model(api_key: str):
    """
    Determine which client and model to use based on the API key
    Returns (client, model_id, pro_model_id)
    """
    # Check if it's a Gemini API key (starts with 'AIza' typically)
    if api_key and (api_key.startswith('AIza') or 'gemini' in api_key.lower()):
        # Use Gemini client
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        # Allow environment variables to override default models
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        gemini_pro_model = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
        return client, gemini_model, gemini_pro_model
    else:
        # Use OpenAI client
        client = OpenAI(api_key=api_key)
        # Allow environment variables to override default models
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        openai_pro_model = os.getenv("OPENAI_PRO_MODEL", "gpt-4o")
        return client, openai_model, openai_pro_model

# Default client setup (will be overridden in functions)
if gemini_api_key:
    client, MODEL_ID, PRO_MODEL_ID = get_client_and_model(gemini_api_key)
elif openai_api_key:
    client, MODEL_ID, PRO_MODEL_ID = get_client_and_model(openai_api_key)
else:
    # Fallback setup
    client = OpenAI()
    MODEL_ID = "gpt-4o"
    PRO_MODEL_ID = "gpt-4o"

def get_system_prompt():
    """Get the system prompt with tools included"""
    tools = get_openai_tools_for_icd10()
    return EXTRACT_ICD10_SYSTEM_PROMPT.substitute(tools=tools)

def get_response(transcript: str, messages: List[Dict], api_key: str, system_prompt: str = None):
    """Get response from API - works with both OpenAI and Gemini"""
    
    # Get appropriate client and model for this API key
    client, model_id, _ = get_client_and_model(api_key)
    
    if len(messages) == 0:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the transcript: {transcript}\n\nWhat's your first step?"}
        ]

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    return response, messages

def preprocess_transcript(transcript: str, api_key: str):
    """
    First step: Extract relevant medical details and remove fluff from transcript
    Works with both OpenAI and Gemini APIs
    """
    
    # Get appropriate client and model for this API key
    client, model_id, _ = get_client_and_model(api_key)
    
    messages = [
        {"role": "system", "content": PREPROCESSING_PROMPT},
        {"role": "user", "content": f"Please extract the relevant medical information from this transcript and remove all fluff:\n\n{transcript}"}
    ]
    
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    print(f"Response: {response}")
    cleaned_transcript = response.choices[0].message.content
    return cleaned_transcript

def add_confidence_scores(chat_history: List[Dict], entries: List[Dict], api_key: str):
    """
    Confidence scoring function that works with both OpenAI and Gemini APIs
    """
    if not entries:
        return entries
    
    # Get appropriate client and models for this API key
    client, _, pro_model_id = get_client_and_model(api_key)
    
    # Format the chat history and entries for review
    chat_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    # Format entries for review
    entries_for_review = []
    for i, entry in enumerate(entries):
        entries_for_review.append({
            "icd10_code": entry.get('icd10_code', 'Unknown'),
            "icd10_condition_name": entry.get('icd10_condition_name', 'Unknown'),
            "reasoning": entry.get('reasoning', 'No reasoning provided')
        })
    
    # Simple confidence scoring prompt
    prompt = f"""Based on the chat history and extracted ICD-10 entries, assess the confidence level for each entry.

Chat History:
{chat_summary}

Entries to Review:
{json.dumps(entries_for_review, indent=2)}

For each entry, provide:
- icd10_code: the icd10 code of the entry
- confidence_reasoning: brief explanation for why this rating is being given
- confidence: either "confident" or "requires_human_review"  

Return as JSON array with format:
[{{"icd10_code": "A00.0", "confidence_reasoning": "Clear evidence in transcript", "confidence": "confident"}}]"""

    response = client.chat.completions.create(
        model=pro_model_id,
        messages=[
            {"role": "system", "content": "You are a medical coding quality reviewer. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    # Parse response
    response_content = response.choices[0].message.content
    confidence_data = json.loads(response_content)
    
    # Handle both list and dict responses
    if isinstance(confidence_data, dict):
        # If it's a dict, look for common keys that contain the assessments
        if 'results' in confidence_data:
            assessments = confidence_data['results']
        elif 'assessments' in confidence_data:
            assessments = confidence_data['assessments']
        elif 'entries' in confidence_data:
            assessments = confidence_data['entries']
        elif 'confidence_scores' in confidence_data:
            assessments = confidence_data['confidence_scores']
        else:
            # Single entry case
            assessments = [confidence_data]
    else:
        assessments = confidence_data
    
    # Apply confidence scores to entries by matching ICD codes
    for entry in entries:
        entry_code = entry.get('icd10_code', '')
        
        # Find matching assessment by ICD code
        matching_assessment = None
        for assessment in assessments:
            if assessment.get('icd10_code') == entry_code:
                matching_assessment = assessment
                break
        
        if matching_assessment:
            entry['confidence'] = matching_assessment.get('confidence', 'requires_human_review')
            entry['confidence_reasoning'] = matching_assessment.get('confidence_reasoning', 'No reasoning provided')
        else:
            # Fallback if no match found
            entry['confidence'] = 'requires_human_review'
            entry['confidence_reasoning'] = 'No matching confidence assessment found'
    
    return entries

def get_icd10_codes_with_chat_history(transcript: str, api_key: str, max_iterations: int = MAX_TURNS):
    """
    Modified version that returns both entries and chat history
    Simple version from notebook that works
    """
    counter = 0
    messages = []
    context_vars = ICD10ContextVariables(
        xml_file_path_alphabetical=ALPHABETICAL_INDEX_PATH,
        xml_file_path_tabular=TABULAR_LIST_PATH,
        entries=[]
    )
    
    while counter < max_iterations:
        try:
            response, messages = get_response(transcript, messages, api_key, system_prompt=get_system_prompt())
            text = response.choices[0].message.content
            
            if "stop" in text.lower():
                break
                
            print(f"Step {counter + 1}: {text[:100]}...")  # Print first 100 chars for debugging
            
            tool_call = extract_tool_call_from_response(text)
            
            if tool_call:
                result = execute_function_with_context(
                    tool_call["function_name"], 
                    tool_call["arguments"], 
                    context_vars
                )
                print(f"Tool result: {str(result)[:100]}...")  # Print first 100 chars for debugging
                tool_call_message = {"role": "user", "content": f"Function returned: {result}. What's your next step?"}
            else:
                tool_call_message = {"role": "user", "content": "No function call found. What's your next step?"}

            messages += [
                {"role": "assistant", "content": text},
                tool_call_message
            ]
            
            counter += 1
            
        except Exception as e:
            print(f"Error in step {counter + 1}: {e}")
            break

    return context_vars.entries, messages

def get_icd10_codes_with_full_pipeline(transcript: str, api_key: str, max_iterations: int = MAX_TURNS):
    """
    Complete pipeline: preprocessing -> extraction -> confidence scoring
    Simple version from notebook that works
    """
    # Step 1: Preprocess transcript
    print("Step 1: Preprocessing transcript to extract relevant medical information...")
    cleaned_transcript = preprocess_transcript(transcript, api_key)
    print(f"Cleaned transcript: {cleaned_transcript[:200]}...")  # Print first 200 chars
    print("\n" + "="*50 + "\n")
    
    # Step 2: Extract ICD-10 codes (modified to return chat history too)
    print("Step 2: Extracting ICD-10 codes from cleaned transcript...")
    entries, chat_history = get_icd10_codes_with_chat_history(cleaned_transcript, api_key, max_iterations)
    print(f"Extracted {len(entries)} entries")
    print("\n" + "="*50 + "\n")
    
    # Step 3: Add confidence scores
    print("Step 3: Adding confidence scores to entries...")
    scored_entries = add_confidence_scores(chat_history, entries, api_key)
    print(f"Added confidence scores to {len(scored_entries)} entries")
    
    return scored_entries, chat_history

def extract_confidence_summary(entries: List[Dict]) -> str:
    """
    Extract confidence summary from entries for storage in DataFrame
    """
    if not entries:
        return "no_entries"
    
    confident_count = sum(1 for entry in entries if entry.get('confidence') == 'confident')
    review_count = sum(1 for entry in entries if entry.get('confidence') == 'requires_human_review')
    total_count = len(entries)
    
    return f"{confident_count}_confident_{review_count}_review_{total_count}_total" 