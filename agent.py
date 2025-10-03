"""
Starter scaffold for the CS 294-264 HW1 ReAct agent.

Students must implement a minimal ReAct agent that:
- Maintains a message history tree (role, content, timestamp, unique_id, parent, children)
- Uses a textual function-call format (see ResponseParser) with rfind-based parsing
- Alternates Reasoning and Acting until calling the tool `finish`
- Supports tools: `run_bash_cmd`, `finish`, and `add_instructions_and_backtrack`

This file intentionally omits core implementations and replaces them with
clear specifications and TODOs.
"""

from typing import List, Callable, Dict, Any
from datetime import datetime
import time

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect

class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history tree with unique ids
    - Builds the LLM context from the root to current node
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
        self.name: str = name
        self.parser = parser
        self.llm = llm

        # Message tree storage
        self.id_to_message: List[Dict[str, Any]] = []
        self.root_message_id: int = -1
        self.current_message_id: int = -1

        # Registered tools
        self.function_map: Dict[str, Callable] = {}

        # Set up the initial structure of the history
        # Create required root nodes and a user node (task) and an instruction node.
        system_prompt = """You are a Smart ReAct agent solving software engineering tasks.

CRITICAL RULES:
1. You MUST use the available tools to ACTUALLY EDIT FILES in the codebase.
2. NEVER just describe what changes should be made - you must EXECUTE the changes using tools.
3. DO NOT write pseudo-patches or text descriptions of changes - USE THE TOOLS to modify files.
4. After making changes, verify them by reading the files back.
5. Only call 'finish' after you have actually modified the necessary files.

WORKFLOW:
1. Understand the problem by reading relevant files and exploring the codebase
2. Identify which files need to be changed
3. USE run_bash_cmd or replace_in_file to ACTUALLY MODIFY the files
4. Verify your changes by reading the modified files
5. Call finish() with a brief summary only after files are modified

TOOL USAGE EXAMPLES:
- To view a file: run_bash_cmd(command="cat path/to/file.py")
- To edit a file: replace_in_file(file_path="path/to/file.py", from_line=10, to_line=15, content="new code here")
- To use sed: run_bash_cmd(command="sed -i 's/old/new/g' path/to/file.py")
- To verify changes: run_bash_cmd(command="git diff path/to/file.py")

REMEMBER: Your job is to make ACTUAL file modifications, not write descriptions!"""
        self.system_message_id = self.add_message("system", system_prompt)
        self.user_message_id = self.add_message("user", "")
        
        # Add default instructions to emphasize the importance of actual file edits
        default_instructions = """Follow the workflow strictly:
1. Explore the codebase to understand the problem
2. Identify exact files and lines that need changes
3. Use run_bash_cmd or replace_in_file to MAKE THE ACTUAL CHANGES
4. Verify your edits worked (git diff, cat the file, etc.)
5. Only then call finish()

DO NOT just describe changes - you must execute them!"""
        self.instructions_message_id = self.add_message("instructor", default_instructions)
        
        # NOTE: mandatory finish function that terminates the agent
        self.add_functions([self.finish])

    # -------------------- MESSAGE TREE --------------------
    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the tree.

        The message must include fields: role, content, timestamp, unique_id, parent, children.
        Maintain a pointer to the current node and the root node.
        """
        # Create the new message with all required fields
        message_id = len(self.id_to_message)
        message = {
            "role": role,
            "content": content,
            "timestamp": int(time.time()),  # Use Unix timestamp (int) for JSON serialization
            "unique_id": message_id,
            "parent": self.current_message_id,
            "children": []
        }
        
        # Add the message to the tree
        self.id_to_message.append(message)
        
        # If there's a parent, link this message as its child
        if self.current_message_id != -1:
            self.id_to_message[self.current_message_id]["children"].append(message_id)
        
        # Update the current message pointer
        self.current_message_id = message_id
        
        # If this is the first message, set it as the root
        if self.root_message_id == -1:
            self.root_message_id = message_id
        
        return message_id

    def set_message_content(self, message_id: int, content: str) -> None:
        """Update message content by id."""
        self.id_to_message[message_id]["content"] = content

    def get_context(self) -> str:
        """
        Build the full LLM context by walking from the root to the current message.
        """
        # Walk from current message back to root to build the path
        path = []
        message_id = self.current_message_id
        
        while message_id != -1:
            path.append(message_id)
            message_id = self.id_to_message[message_id]["parent"]
        
        # Reverse the path to get root-to-current order
        path.reverse()
        
        # Build the context string by concatenating all messages
        context_parts = []
        for msg_id in path:
            context_parts.append(self.message_id_to_context(msg_id))
        
        return "".join(context_parts)

    # -------------------- REQUIRED TOOLS --------------------
    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        """
        # Register each tool in the function map using the function name as the key
        for tool in tools:
            self.function_map[tool.__name__] = tool
    
    def finish(self, result: str):
        """Call this function ONLY AFTER you have ACTUALLY MODIFIED the necessary files using run_bash_cmd or replace_in_file.
        
        This function generates a git diff patch of your changes. If you haven't edited any files yet, 
        the patch will be empty and your solution will fail.
        
        BEFORE calling finish:
        1. Make sure you've used run_bash_cmd or replace_in_file to EDIT files
        2. Verify your changes with: run_bash_cmd(command="git diff")
        3. Confirm files were modified: run_bash_cmd(command="git status")

        Args: 
            result (str): A brief summary of what you changed (the actual patch is generated automatically)

        Returns:
            The result passed as an argument. The result is then returned by the agent's run method.
        """
        return result 

    def add_instructions_and_backtrack(self, instructions: str, at_message_id: int):
        """
        The agent should call this function if it is making too many mistakes or is stuck.

        The function changes the content of the instruction node with 'instructions' and
        backtracks at the node with id 'at_message_id'. Backtracking means the current node
        pointer moves to the specified node and subsequent context is rebuilt from there.

        Returns a short success string.
        """
        # Update the instruction node content
        self.set_message_content(self.instructions_message_id, instructions)
        
        # Backtrack: move the current pointer to the specified message
        self.current_message_id = at_message_id
        
        return f"Instructions updated and backtracked to message {at_message_id}."

    # -------------------- MAIN LOOP --------------------
    def run(self, task: str, max_steps: int) -> str:
        """
        Run the agent's main ReAct loop:
        - Set the user prompt
        - Loop up to max_steps (<= 100):
            - Build context from the message tree
            - Query the LLM
            - Parse a single function call at the end (see ResponseParser)
            - Execute the tool
            - Append tool result to the tree
            - If `finish` is called, return the final result
        """
        # Set the user message content with the task
        self.set_message_content(self.user_message_id, task)
        
        # Main ReAct loop
        for step in range(max_steps):
            try:
                # Build context from the message tree
                context = self.get_context()
                
                # Query the LLM
                response = self.llm.generate(context)
                
                # Add the assistant's response to the message tree
                self.add_message("assistant", response)
                
                # Parse the function call from the response
                parsed = self.parser.parse(response)
                function_name = parsed["name"]
                arguments = parsed["arguments"]
                
                # Check if the function exists in the function map
                if function_name not in self.function_map:
                    error_msg = f"Error: Function '{function_name}' not found in available tools."
                    self.add_message("system", error_msg)
                    continue
                
                # Execute the tool
                func = self.function_map[function_name]
                try:
                    result = func(**arguments)
                    
                    # If finish was called, return the result
                    if function_name == "finish":
                        return result
                    
                    # Add the tool result to the message tree
                    result_msg = f"Tool '{function_name}' executed successfully. Result:\n{result}"
                    self.add_message("system", result_msg)
                    
                except Exception as e:
                    # Handle tool execution errors
                    error_msg = f"Error executing tool '{function_name}': {str(e)}"
                    self.add_message("system", error_msg)
                    
            except Exception as e:
                # Handle parsing or other errors
                error_msg = f"Error in agent loop: {str(e)}"
                self.add_message("system", error_msg)
        
        # If we reach max_steps without calling finish
        return f"Agent reached maximum steps ({max_steps}) without completing the task."

    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        message = self.id_to_message[message_id]
        header = f'----------------------------\n|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        content = message["content"]
        if message["role"] == "system":
            tool_descriptions = []
            for tool in self.function_map.values():
                signature = inspect.signature(tool)
                docstring = inspect.getdoc(tool)
                tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                tool_descriptions.append(tool_description)

            tool_descriptions = "\n".join(tool_descriptions)
            return (
                f"{header}{content}\n"
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
            )
        elif message["role"] == "instructor":
            return f"{header}YOU MUST FOLLOW THE FOLLOWING INSTRUCTIONS AT ANY COST. OTHERWISE, YOU WILL BE DECOMISSIONED.\n{content}\n"
        else:
            return f"{header}{content}\n"

def main():
    from envs import DumbEnvironment
    llm = OpenAIModel("----END_FUNCTION_CALL----", "gpt-5-mini")
    print(llm.generate("What is the capital of France?"))
    parser = ResponseParser()

    env = DumbEnvironment()
    dumb_agent = ReactAgent("dumb-agent", parser, llm)
    dumb_agent.add_functions([env.run_bash_cmd])
    result = dumb_agent.run("Show the contents of all files in the current directory.", max_steps=10)
    print(result)

if __name__ == "__main__":
    # Optional: students can add their own quick manual test here.
    main()