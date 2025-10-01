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
arg1_name
arg1_value (can be multiline)
{ARG_SEP}
arg2_name
arg2_value (can be multiline)
...
{END_CALL}
"""

    def parse(self, text: str) -> dict:
        """
        Parse the function call from `text` using string.rfind to avoid confusion with
        earlier delimiter-like content in the reasoning.

        Returns a dictionary: {"thought": str, "name": str, "arguments": dict}
        """
        # TODO(student): Implement rfind-based parsing per the assignment description.
        # Hints:
        # - Find END_CALL via rfind; then find the matching BEGIN_CALL before it via rfind
        # - Everything before BEGIN_CALL is the model's thought
        # - Between BEGIN_CALL and END_CALL: first block is function name, subsequent blocks
        #   are argument name/value pairs separated by ARG_SEP, values may be multiline
        # - Raise ValueError on malformed inputs
        raise NotImplementedError("ResponseParser.parse must be implemented by the student")
