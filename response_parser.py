class ResponseParser:
    """
    Parses LLM responses to extract a single function call using a rigid textual format.

    The LLM must output exactly one function call at the end of its response.
    Do NOT use JSON or XML. Use rfind to locate the final markers.
    """

    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL = "----END_FUNCTION_CALL----"
    ARG_SEP = "----ARG----"

    # Students should include this exact template in the system prompt so the LLM follows it.
    response_format = f"""
your_thoughts_here
...
{BEGIN_CALL}
function_name
{ARG_SEP}
param_name_1
param_value_1 (can be multiline, NO {ARG_SEP} separator between name and value!)
{ARG_SEP}
param_name_2
param_value_2 (can be multiline)
{END_CALL}

IMPORTANT: Each argument block starts with {ARG_SEP}, then the parameter NAME on the first line, 
then the parameter VALUE on subsequent lines. Do NOT put {ARG_SEP} between name and value!

Example:
{BEGIN_CALL}
run_bash_cmd
{ARG_SEP}
command
ls -la
{END_CALL}
"""

    def parse(self, text: str) -> dict:
        """
        Parse the function call from `text` using string.rfind to avoid confusion with
        earlier delimiter-like content in the reasoning.

        Returns a dictionary: {"thought": str, "name": str, "arguments": dict}
        """
        # Find the last occurrence of END_CALL
        end_pos = text.rfind(self.END_CALL)
        if end_pos == -1:
            raise ValueError(f"Missing {self.END_CALL} marker in response")
        
        # Find the last occurrence of BEGIN_CALL before END_CALL
        begin_pos = text.rfind(self.BEGIN_CALL, 0, end_pos)
        if begin_pos == -1:
            raise ValueError(f"Missing {self.BEGIN_CALL} marker in response")
        
        # Extract the thought (everything before BEGIN_CALL)
        thought = text[:begin_pos].strip()
        
        # Extract the function call body (between BEGIN_CALL and END_CALL)
        call_start = begin_pos + len(self.BEGIN_CALL)
        call_body = text[call_start:end_pos].strip()
        
        # Split by ARG_SEP to get function name and arguments
        parts = call_body.split(self.ARG_SEP)
        
        if len(parts) < 1:
            raise ValueError("Function call body is empty")
        
        # First part is the function name
        function_name = parts[0].strip()
        if not function_name:
            raise ValueError("Function name is empty")
        
        # Parse arguments (pairs of name and value)
        arguments = {}
        # Arguments come in pairs after the function name
        for i in range(1, len(parts)):
            arg_block = parts[i].strip()
            if not arg_block:
                continue
            
            # Split into name and value (first line is name, rest is value)
            lines = arg_block.split('\n', 1)
            arg_name = lines[0].strip()
            # Only strip trailing whitespace from value to preserve leading indentation
            arg_value = lines[1].rstrip() if len(lines) > 1 else ""
            
            if arg_name:
                arguments[arg_name] = arg_value
        
        return {
            "thought": thought,
            "name": function_name,
            "arguments": arguments
        }
