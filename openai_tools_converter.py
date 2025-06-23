import inspect
import json
import re
import ast
from typing import List, Dict, Any, Callable, Optional
from pydantic import BaseModel, Field
from simple_xml_lookup import lookup_alphabetical_index, lookup_tabular_list


class ICD10ContextVariables(BaseModel):
    """
    Context variables for ICD-10 coding functions that are handled separately from the model.
    """
    xml_file_path_alphabetical: Optional[str] = Field(
        default=None, 
        description="Path to the ICD-10 alphabetical index XML file"
    )
    xml_file_path_tabular: Optional[str] = Field(
        default=None, 
        description="Path to the ICD-10 tabular list XML file"
    )
    xml_string: Optional[str] = Field(
        default=None, 
        description="XML content as string (alternative to file path)"
    )
    entries: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of previous ICD-10 suspect entries (conversation history)"
    )
    
    def get_xml_file_path_for_function(self, function_name: str) -> Optional[str]:
        """
        Get the appropriate XML file path based on the function being called.
        
        Args:
            function_name (str): Name of the function being called
            
        Returns:
            Optional[str]: The appropriate XML file path
        """
        if function_name == "lookup_alphabetical_index":
            return self.xml_file_path_alphabetical
        elif function_name == "lookup_tabular_list":
            return self.xml_file_path_tabular
        return None


def convert_tools_to_openai_format(tools: List[Callable]) -> List[Dict[str, Any]]:
    """
    Convert a list of Python functions to OpenAI tools format.
    
    Args:
        tools (List[Callable]): List of Python functions to convert
        
    Returns:
        List[Dict[str, Any]]: List of tools in OpenAI format
    """
    openai_tools = []
    
    for tool in tools:
        if not callable(tool):
            continue
            
        # Get function metadata
        func_name = tool.__name__
        func_doc = inspect.getdoc(tool) or f"Function {func_name}"
        func_signature = inspect.signature(tool)
        
        # Build parameters schema
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Context variables that should be handled separately (not shown to model)
        context_params = {
            'xml_file_path', 'xml_string', 'entries'
        }
        
        # Process each parameter
        for param_name, param in func_signature.parameters.items():
            # Skip context variables - these will be handled separately
            if param_name in context_params:
                continue
                
            param_schema = _get_parameter_schema(param_name, param, func_name)
            parameters["properties"][param_name] = param_schema
            
            # Add to required if no default value and not a context variable
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        # Create OpenAI tool format
        openai_tool = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": func_doc,
                "parameters": parameters
            }
        }
        
        openai_tools.append(openai_tool)
    
    return openai_tools


def _get_parameter_schema(param_name: str, param: inspect.Parameter, func_name: str) -> Dict[str, Any]:
    """
    Convert a function parameter to OpenAI parameter schema.
    
    Args:
        param_name (str): Name of the parameter
        param (inspect.Parameter): Parameter object from function signature
        func_name (str): Name of the function (for context)
        
    Returns:
        Dict[str, Any]: Parameter schema in OpenAI format
    """
    # Default schema
    schema = {"type": "string"}
    
    # Handle specific parameter names and types
    if param_name == "query":
        schema = {
            "type": "string",
            "description": "The medical term or condition to search for in the alphabetical index"
        }
    elif param_name == "code" or param_name == "icd10_code":
        schema = {
            "type": "string", 
            "description": "The ICD-10 code to look up in the tabular list (e.g., 'E11.9', 'I10')"
        }
    elif param_name == "icd10_suspect_entry":
        schema = {
            "type": "object",
            "description": "Dictionary containing ICD-10 coding analysis results",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "The logical reasoning for the selected ICD-10 code"
                },
                "icd10_code": {
                    "type": "string", 
                    "description": "The final ICD-10 code that has been decided on"
                },
                "alternate_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of other codes that might apply if unsure about the main code"
                }
            },
            "required": ["reasoning", "icd10_code"]
        }
    
    # Handle type annotations if available
    if param.annotation != inspect.Parameter.empty:
        if param.annotation == str:
            schema["type"] = "string"
        elif param.annotation == int:
            schema["type"] = "integer"
        elif param.annotation == float:
            schema["type"] = "number"
        elif param.annotation == bool:
            schema["type"] = "boolean"
        elif param.annotation == list:
            schema["type"] = "array"
        elif param.annotation == dict:
            schema["type"] = "object"
    
    return schema


def create_icd10_suspect_entry(icd10_suspect_entry: Dict[str, Any], entries: List[Dict] = None) -> List[Dict]:
    """
    Create an ICD10 suspect entry and add it to the list of entries.
    
    Args:
        icd10_suspect_entry (Dict[str, Any]): Dictionary with reasoning, icd10_code, and optional alternate_codes
        entries (List[Dict], optional): Existing list of entries. Defaults to empty list.
    
    Returns:
        List[Dict]: Updated list of entries including the new suspect entry
    """
    if entries is None:
        entries = []
    
    entries.append(icd10_suspect_entry)
    return entries


def execute_function_with_context(
    function_name: str, 
    function_args: Dict[str, Any], 
    context: ICD10ContextVariables
) -> Any:
    """
    Execute a function with context variables injected.
    
    Args:
        function_name (str): Name of the function to execute
        function_args (Dict[str, Any]): Arguments provided by the model
        context (ICD10ContextVariables): Context variables to inject
        
    Returns:
        Any: Result of the function execution
    """
    # Map function names to actual functions
    function_map = {
        "lookup_alphabetical_index": lookup_alphabetical_index,
        "lookup_tabular_list": lookup_tabular_list,
        "create_icd10_suspect_entry": create_icd10_suspect_entry
    }
    
    if function_name not in function_map:
        raise ValueError(f"Unknown function: {function_name}")
    
    func = function_map[function_name]
    
    # Inject context variables
    if function_name in ["lookup_alphabetical_index", "lookup_tabular_list"]:
        # Add XML file path or string
        xml_file_path = context.get_xml_file_path_for_function(function_name)
        if xml_file_path:
            function_args["xml_file_path"] = xml_file_path
        elif context.xml_string:
            function_args["xml_string"] = context.xml_string
    
    elif function_name == "create_icd10_suspect_entry":
        # Add entries history
        function_args["entries"] = context.entries.copy()
    
    # Execute the function
    result = func(**function_args)
    
    # Update context if needed
    if function_name == "create_icd10_suspect_entry":
        context.entries = result
    
    return result


def get_openai_tools_for_icd10() -> List[Dict[str, Any]]:
    """
    Get the OpenAI tools format for the ICD-10 coding functions.
    
    Returns:
        List[Dict[str, Any]]: List of tools in OpenAI format
    """
    tools = [lookup_alphabetical_index, lookup_tabular_list, create_icd10_suspect_entry]
    return convert_tools_to_openai_format(tools)


def get_context_variables() -> Dict[str, str]:
    """
    Get the context variables that should be handled separately from the model.
    
    Returns:
        Dict[str, str]: Dictionary describing context variables
    """
    return {
        "xml_file_path": "Path to the ICD-10 XML files (alphabetical index and tabular list)",
        "xml_string": "XML content as string (alternative to file path)",
        "entries": "List of previous ICD-10 suspect entries (conversation history)"
    }

def extract_tool_call_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract the tool call from the response and return function name and args.
    Looks for dictionary patterns like:
    {"function_name": "lookup_alphabetical_index", "arguments": {"query": "Diabetes"}}
    """
    import re
    import ast
    
    # Look for dictionary patterns from first { to last } (including multiline)
    dict_pattern = r'\{.*\}'
    
    # Find potential dictionary matches (re.DOTALL makes . match newlines)
    matches = re.findall(dict_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            # Safely evaluate the dictionary string
            tool_call = ast.literal_eval(match)
            
            # Validate it has the expected structure
            if (isinstance(tool_call, dict) and 
                "function_name" in tool_call and 
                "arguments" in tool_call and
                isinstance(tool_call["arguments"], dict)):
                return tool_call
                
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, try json.loads as fallback
            try:
                tool_call = json.loads(match)
                if (isinstance(tool_call, dict) and 
                    "function_name" in tool_call and 
                    "arguments" in tool_call and
                    isinstance(tool_call["arguments"], dict)):
                    return tool_call
            except json.JSONDecodeError:
                continue
    
    return None


def execute_tool_call_from_response(
    response: str, 
    context: ICD10ContextVariables
) -> Any:
    """
    Extract and execute a tool call from a response string.
    
    Args:
        response (str): The response containing the tool call
        context (ICD10ContextVariables): Context variables to inject
        
    Returns:
        Any: Result of the function execution, or None if no tool call found
    """
    tool_call_data = extract_tool_call_from_response(response)
    
    if tool_call_data:
        function_name = tool_call_data["function_name"]
        arguments = tool_call_data["arguments"]
        
        # Execute the function with context
        return execute_function_with_context(function_name, arguments, context)
    
    return None


# Example usage and testing
if __name__ == "__main__":
    # Get OpenAI format tools
    openai_tools = get_openai_tools_for_icd10()
    
    print("OpenAI Tools Format:")
    print(json.dumps(openai_tools, indent=2))
    
    print("\n" + "="*50)
    print("Context Variables (handled separately):")
    context_vars = get_context_variables()
    for var, description in context_vars.items():
        print(f"- {var}: {description}")
    
    print("\n" + "="*50)
    print("Example Context Variables Model:")
    context = ICD10ContextVariables(
        xml_file_path_alphabetical="data/icd10cm_index_2025.xml",
        xml_file_path_tabular="data/icd10cm_tabular_2025.xml"
    )
    print(context.model_dump_json(indent=2))
    
    print("\n" + "="*50)
    print("Example Tool Call Extraction and Execution:")
    
    # Example response with tool call
    example_response = '''Following the guidelines, I will search for "Diabetes" first.

**Tool Call:**
```python
print(lookup_alphabetical_index(query="Diabetes"))
```'''
    
    print("Original response:")
    print(example_response)
    
    # Extract the tool call
    extracted_call = extract_tool_call_from_response(example_response)
    print(f"\nExtracted tool call: {extracted_call}")
    
    # Note: You would execute it like this (but we can't run it without the XML files):
    # result = execute_tool_call_from_response(example_response, context)
    # print(f"Result: {result}")
    
    print("\nTo execute the tool call, use:")
    print("result = execute_tool_call_from_response(response_string, context)")
    print("or")
    print("result = parse_and_execute_tool_call(extracted_call, context)")