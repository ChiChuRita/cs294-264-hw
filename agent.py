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
import re

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect


# ============================================================================
# V3 VALIDATION FUNCTIONS - Prevent catastrophic edits
# ============================================================================

def validate_edit_safety(file_path: str, from_line: int, to_line: int, 
                        old_content: str, new_content: str) -> None:
    """
    Prevent accidental deletion of class/function definitions.
    Raises ValueError if edit would delete critical code structures.
    """
    # Patterns that indicate critical code structure
    critical_patterns = [
        (r'^\s*class\s+\w+', 'class definition'),
        (r'^\s*def\s+\w+', 'function definition'),
        (r'^\s*async\s+def\s+\w+', 'async function definition'),
    ]
    
    # Check if old content has critical patterns that new content doesn't
    for pattern, description in critical_patterns:
        old_matches = re.findall(pattern, old_content, re.MULTILINE)
        new_matches = re.findall(pattern, new_content, re.MULTILINE)
        
        if len(old_matches) > len(new_matches):
            raise ValueError(
                f"⛔ BLOCKED: This edit would DELETE a {description}!\n"
                f"File: {file_path}, Lines: {from_line}-{to_line}\n"
                f"Found in old: {old_matches}\n"
                f"Found in new: {new_matches}\n"
                f"\n"
                f"This is almost always a mistake!\n"
                f"COMMON CAUSES:\n"
                f"- Wrong line numbers (e.g., starting at class line instead of method line)\n"
                f"- Forgot to include the class/function definition in new_content\n"
                f"\n"
                f"TO FIX:\n"
                f"- Use 'sed -n {from_line},{to_line}p {file_path}' to see exact lines\n"
                f"- Adjust from_line to START AFTER the definition line\n"
                f"- Or include the definition in new_content if you're refactoring\n"
            )


def enforce_comprehensive_testing(agent_messages: list) -> None:
    """
    Verify agent ran comprehensive tests before allowing finish().
    Raises ValueError if testing requirements not met.
    """
    # Extract all assistant messages that called run_tests
    test_calls = []
    for msg in agent_messages:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if 'run_tests' in content or ('run_bash_cmd' in content and 'pytest' in content):
                test_calls.append(content)
    
    if len(test_calls) < 1:
        raise ValueError(
            "❌ CANNOT FINISH: No test execution detected!\n"
            "\n"
            "REQUIRED before calling finish():\n"
            "1. Run originally failing tests to verify your fix works\n"
            "2. Run ENTIRE test file/module to check for regressions\n"
            "\n"
            "Example:\n"
            "  run_tests(test_path='tests/test_views.py::TestClass::test_failing')\n"
            "  run_tests(test_path='tests/test_views.py')  # ALL tests in file\n"
            "\n"
            f"Currently you have: {len(test_calls)} test run(s)\n"
        )
    
    # Check if comprehensive testing (not just single test)
    comprehensive_test = False
    for call in test_calls:
        # Comprehensive if: no "::" in path (full file), or explicit mention of "all"
        if ('run_tests' in call and '::' not in call) or 'all' in call.lower():
            comprehensive_test = True
            break
    
    if not comprehensive_test:
        raise ValueError(
            "❌ CANNOT FINISH: Only ran specific tests, not comprehensive!\n"
            "\n"
            "You MUST run the ENTIRE test file/module to catch regressions.\n"
            "\n"
            "What you did:\n"
            f"  {test_calls[-1] if test_calls else 'Nothing'}  # Only specific test(s)\n"
            "\n"
            "What you MUST do:\n"
            "  run_tests(test_path='tests/full_test_file.py')  # ALL tests\n"
            "\n"
            "This prevents breaking existing tests (PASS_TO_PASS failures).\n"
        )

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
        system_prompt = """You are a ReAct agent solving software engineering tasks by EDITING source code files.

**CRITICAL RULES:**
1. NEVER edit test files - they define requirements. Edit SOURCE CODE to pass tests.
2. Make ACTUAL file changes using replace_in_file/search_and_replace - not just comments!
3. Call ONE tool per response. Wait for results before next action.
4. Preserve indentation exactly - use sed to inspect before editing.
5. Run comprehensive tests before finish(): failing tests + ENTIRE test file/module.

**WORKFLOW:**
1. Read failing tests to understand requirements
2. Make targeted code changes (not test changes!)
3. Verify each edit: syntax check, git diff
4. Run tests: specific failures, then full module
5. Fix any issues, re-test comprehensively
6. Call finish() only when all tests pass

**COMMON MISTAKES TO AVOID:**
- Modifying test_*.py files (❌ Wrong! Edit source code instead)
- Adding only comments without logic changes (❌ Not a fix!)
- Broken syntax from orphaned code/conditions (❌ Always verify!)
- Skipping comprehensive testing (❌ Causes regressions!)

Your edits must change BEHAVIOR, not just add documentation."""

        self.system_message_id = self.add_message("system", system_prompt)
        self.user_message_id = self.add_message("user", "")
        
        # Add default instructions to emphasize the importance of actual file edits
        default_instructions = """MANDATORY WORKFLOW:

1. **READ TESTS FIRST**: Locate and read ALL failing tests to understand requirements
   - Never modify test files - they define what's correct
   - Tests tell you WHAT needs to work, not HOW to implement it

2. **IDENTIFY SOURCE CODE**: Find the actual implementation files (NOT test files!)
   - Common locations: src/, lib/, main package directories
   - Avoid: tests/, test_*.py, *_test.py

3. **MAKE REAL CODE CHANGES**: Edit source files to fix the issue
   - Use replace_in_file or search_and_replace
   - Add/modify LOGIC, not just comments
   - Preserve exact indentation (use sed first to check)

4. **VERIFY IMMEDIATELY**: After each edit, run:
   - check_python_syntax (catch syntax errors early)
   - git diff (confirm actual changes)
   - wc -l (ensure file not corrupted)

5. **TEST COMPREHENSIVELY** (MANDATORY before finish):
   - Run failing tests: run_tests(test_path='path/to/test::specific_test')
   - Run ENTIRE module: run_tests(test_path='path/to/test_file.py')
   - Both must pass - partial success is not enough!

6. **FIX ISSUES**: If tests fail, analyze error and iterate
   - Don't guess - read the actual error message
   - Check for edge cases (None, empty, zero)
   - Ensure you modified source code, not tests

7. **FINISH**: Only after ALL tests pass

CRITICAL: One tool call per response. Changes must modify BEHAVIOR, not just add comments."""
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
        4. Run ALL tests to ensure PASS_TO_PASS tests still pass (no regressions)

        Args: 
            result (str): A brief summary of what you changed (the actual patch is generated automatically)

        Returns:
            The result passed as an argument. The result is then returned by the agent's run method.
        """
        import subprocess
        
        # VALIDATION 0: Enforce comprehensive testing (V3 FIX)
        try:
            enforce_comprehensive_testing(self.id_to_message)
        except ValueError as e:
            # Re-raise as system message so agent sees the error
            self.add_message("system", str(e))
            raise
        
        # VALIDATION 1: Check for Python syntax errors in modified files
        try:
            # Get list of modified Python files
            status_result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            modified_files = [
                f for f in status_result.stdout.strip().split('\n') 
                if f.endswith('.py') and f
            ]
            
            # Check syntax for each modified Python file
            syntax_errors = []
            for file_path in modified_files:
                try:
                    import ast
                    with open(file_path, 'r') as f:
                        ast.parse(f.read(), filename=file_path)
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path}:{e.lineno}: {e.msg}")
                except FileNotFoundError:
                    pass  # File was deleted, skip
            
            if syntax_errors:
                error_msg = (
                    "❌ CANNOT FINISH: Python syntax errors detected!\n\n" +
                    "\n".join(syntax_errors) +
                    "\n\nFix these syntax errors before calling finish()."
                )
                raise ValueError(error_msg)
                
        except subprocess.TimeoutExpired:
            pass  # If git commands timeout, proceed anyway
        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            # Log other errors but don't block
            print(f"Syntax validation warning: {e}")
        
        # VALIDATION 2: Check if there are any changes (patch validation)
        try:
            diff_result = subprocess.run(
                ["git", "diff", "--staged"],
                capture_output=True,
                text=True,
                timeout=5
            )
            unstaged_diff = subprocess.run(
                ["git", "diff"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            patch = diff_result.stdout + unstaged_diff.stdout
            
            # Check if patch is empty or too short
            if not patch or len(patch.strip()) < 10:
                raise ValueError(
                    "❌ CANNOT FINISH: No changes detected!\n"
                    "You must ACTUALLY edit files before calling finish().\n"
                    "Use 'git diff' to verify you have made changes."
                )
            
            # Check if patch contains only deletions (possible file corruption)
            if patch.count('\n-') > patch.count('\n+') * 3:
                raise ValueError(
                    "⚠️ WARNING: Patch contains mostly deletions!\n"
                    "This might indicate file corruption. Review with 'git diff'."
                )
                
        except subprocess.TimeoutExpired:
            pass  # If git commands timeout, proceed anyway
        except Exception as e:
            # Log validation error but don't block finish
            print(f"Patch validation warning: {e}")
        
        # VALIDATION 2: Check for regression testing evidence
        # Look for run_tests calls in recent messages
        recent_messages = self.id_to_message[-20:]  # Check last 20 messages
        test_commands = [
            msg.get("content", "") 
            for msg in recent_messages 
            if msg.get("role") == "assistant"
        ]
        test_commands_str = " ".join(test_commands)
        
        # Check if agent ran comprehensive tests (not just single tests)
        has_run_tests = "run_tests" in test_commands_str or "pytest" in test_commands_str
        has_comprehensive_test = (
            "test_" in test_commands_str and 
            (".py" in test_commands_str or "tests/" in test_commands_str)
        )
        
        if not (has_run_tests and has_comprehensive_test):
            # Issue a strong warning but don't block (agent might have good reason)
            warning_msg = (
                "\n⚠️⚠️⚠️ WARNING: No evidence of comprehensive testing! ⚠️⚠️⚠️\n"
                "You SHOULD have run the ENTIRE test file/module to check for regressions.\n"
                "If you haven't done this, your changes might break existing tests (PASS_TO_PASS).\n"
                "Consider running: run_tests(test_path='tests/full_module_test.py')\n"
            )
            # Add warning to result but allow finish
            result = result + warning_msg
        
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
                    
                    # Add reflection prompt with specific indentation guidance
                    error_str = str(e)
                    is_indentation_error = "IndentationError" in error_str or "unexpected indent" in error_str
                    is_test_failure = "test" in function_name.lower() or "FAILED" in error_str or "AssertionError" in error_str
                    
                    reflection_prompt = f"""The tool call `{function_name}(**{arguments})` failed.

**Error:**
{error_str}

**Reflection:**
1.  **Analyze**: What was the root cause of the error? (e.g., incorrect file path, invalid arguments, syntax error in command)
2.  **Rethink**: How can you achieve the original goal differently? Is there a better tool or a different command?
3.  **Plan**: Formulate your next step based on your analysis.
"""
                    
                    # Add specific guidance for indentation errors
                    if is_indentation_error and function_name == "check_python_syntax":
                        reflection_prompt += """
**INDENTATION ERROR DETECTED!**
This is likely because your `replace_in_file` call didn't preserve the original indentation.

**Recovery Steps:**
1. Use `run_bash_cmd(command="git diff")` to see what changed
2. Identify the indentation mismatch (missing leading spaces/tabs)
3. Use `run_bash_cmd(command="git checkout -- <file>")` to revert the file
4. Re-read the original lines with `run_bash_cmd(command="sed -n 'X,Yp' <file>")` to count spaces
5. Try again with correct indentation in the content parameter
"""
                    
                    # Add specific guidance for test failures
                    if is_test_failure:
                        reflection_prompt += """
**TEST FAILURE DETECTED!**
Your changes may have broken something. Analyze carefully:

**Analysis Steps:**
1. READ the actual error message - what exactly failed?
   - AssertionError? What was expected vs actual?
   - AttributeError? Missing method/property?
   - TypeError? Wrong arguments?
2. Is this a REGRESSION (broke an existing test) or still the original failure?
3. Did you handle edge cases? (None, empty string, zero, etc.)
4. Did you run ALL related tests, not just the target test?

**Recovery Options:**
A. If it's the SAME test still failing:
   - Your fix didn't address the root cause
   - Re-read the test to understand what it expects
   - Consider a different approach

B. If it's a DIFFERENT test now failing (regression):
   - Your fix broke something else
   - Run `git diff` to see what changed
   - Consider reverting and making a more targeted fix
   - Or fix the regression while keeping your original fix
"""
                    self.add_message("system", reflection_prompt)

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