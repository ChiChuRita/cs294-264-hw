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


def validate_edit_safety(file_path: str, from_line: int, to_line: int, 
                        old_content: str, new_content: str) -> None:
    """
    Prevent accidental deletion of class/function definitions.
    Raises ValueError if edit would delete critical code structures.
    """
    critical_patterns = [
        (r'^\s*class\s+\w+', 'class definition'),
        (r'^\s*def\s+\w+', 'function definition'),
        (r'^\s*async\s+def\s+\w+', 'async function definition'),
    ]
    
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
    Simple verification: Made changes? Tested them somehow? Good to go.
    """
    test_commands = [
        msg.get('content', '') for msg in agent_messages 
        if msg.get('role') == 'assistant'
    ]
    ran_any_test = any(
        'pytest' in cmd or 'python -c' in cmd or 'python -m' in cmd or 'test' in cmd.lower()
        for cmd in test_commands
    )
    
    test_results = [
        msg.get('content', '') for msg in agent_messages 
        if msg.get('role') == 'system'
    ]
    has_passed = any('PASSED' in result for result in test_results)
    has_output = any('Result:' in result and len(result) > 100 for result in test_results)
    
    pytest_unavailable = any('No module named pytest' in r for r in test_results)
    
    if not pytest_unavailable and ran_any_test and not has_passed:
        raise ValueError(
            "❌ Tests ran but no PASSED found! Verify: grep PASSED /tmp/out.txt"
        )
    
    if not ran_any_test:
        raise ValueError(
            "❌ No testing performed! Run: python -c '<reproduction_case>'"
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
        
        self.consecutive_syntax_errors = 0
        self.consecutive_not_found = 0
        self.recent_actions = []
        self.failed_approaches = []

        system_prompt = """You are an expert software engineer fixing bugs.

# TOOLS (Use specialized tools - they're safer than bash!)
- **show_file**(file_path: str) - Show file with line numbers
- **read_specific_lines**(file_path, start_line, end_line) - Read lines from file
- **replace_in_file**(file_path, from_line, to_line, content) - Replace lines safely
- **insert_lines**(file_path, after_line, content) - Insert new lines
- **check_syntax**(file_path: str) - Check Python syntax
- **find_failing_test**(issue_description: str) - Smart test discovery from description
- **explore_codebase_deeply**(topic: str) - Deep exploration (classes, functions, tests)
- **run_bash_cmd**(command: str) - Execute bash (use others first!)
- **add_instructions_and_backtrack**(instructions, at_message_id) - Try different approach
- **finish**(result: str) - Submit solution (auto-generates git diff)

# SUCCESS PATTERNS (Learn from these!)

✅ EXAMPLE: django__django-11179 (SUCCEEDED - 40% success rate)
1. Found test: grep -rn "test_fast_delete_instance_set_pk_none"
2. Read test code to understand: "After delete, instance.pk should be None"
3. Located implementation: grep -rn "def delete" django/db/models/deletion.py
4. Made MINIMAL 2-line fix: Added setattr(instance, model._meta.pk.attname, None)
5. Verified with: python -m pytest test_name -xvs | grep PASSED ✓
6. Checked regressions: python -m pytest delete/tests.py -v ✓
7. Submitted with finish()

✅ KEY INSIGHT: Minimal changes (1-5 lines) = high success rate
✅ KEY INSIGHT: Always verify "PASSED" explicitly, not just exit code 0
✅ KEY INSIGHT: Run full test file to catch regressions (PASS_TO_PASS failures)

❌ ANTI-PATTERN: Rewriting entire functions → breaks tests, causes regressions
❌ ANTI-PATTERN: Not verifying "PASSED" in test output → submit broken code
❌ ANTI-PATTERN: Modifying test files → NEVER do this! Tests define correctness
❌ ANTI-PATTERN: Getting stuck in loops → use add_instructions_and_backtrack after 3-5 failures

# YOUR MANTRA
Every turn, ask yourself: "What MINIMAL change would make this specific test pass?"

# CORE RULES
1. **Use specialized tools first**: show_file, read_specific_lines, replace_in_file (safer!)
2. **Never modify test files** - they define correctness
3. **Minimal changes**: Add 1-5 lines, don't rewrite functions
4. **Verify everything**: check_syntax → diff → test → check "PASSED"
5. **One tool per turn** - observe before proceeding

# WORKFLOW EXAMPLE
Problem: "Function crashes on None"

1. Find failing test → `run_bash_cmd("grep -rn 'def test_none' tests/")`
2. Read test → `read_specific_lines("tests/test_module.py", 45, 60)`
3. Find implementation → `run_bash_cmd("grep -rn 'def target_function' src/")`
4. Show source file → `show_file("src/module.py")`
5. Make MINIMAL fix → `insert_lines("src/module.py", 46, "    if value is None: return None")`
6. Check syntax → `check_syntax("src/module.py")`
7. Review changes → `run_bash_cmd("git diff src/module.py")`
8. Run test → `run_bash_cmd("python -m pytest tests/test_module.py::test_none -xvs 2>&1 | tee /tmp/out.txt")`
9. VERIFY "PASSED" → `run_bash_cmd("grep PASSED /tmp/out.txt")`
10. Check regressions → `run_bash_cmd("python -m pytest tests/test_module.py -v")`

# CRITICAL: MINIMAL CHANGES
BAD: Rewriting entire function (20+ lines) → breaks tests, causes regressions
GOOD: Add 1-5 lines only → safer, easier to verify

# TEST VERIFICATION
REQUIRED before finish():
```bash
# Must see "PASSED" explicitly
python -m pytest test::name -xvs 2>&1 | tee /tmp/test.txt
grep "PASSED" /tmp/test.txt  # Must succeed!

# Must check NO regressions  
python -m pytest test_file.py -v | grep -E "failed|ERROR"  # Must be empty!
```

# WHEN TO BACKTRACK
After 5 failed attempts OR stuck in loop:
```
add_instructions_and_backtrack(
    instructions="New approach: [describe different strategy]",
    at_message_id=15
)
```

# RESPONSE FORMAT
EVERY response MUST end with ONE function call:
```
I'll check the test file.
----BEGIN_FUNCTION_CALL----
run_bash_cmd
----ARG----
command
cat tests/test_example.py | head -30
----END_FUNCTION_CALL----
```

NEVER write without calling a tool. NEVER ask user for input."""

        self.system_message_id = self.add_message("system", system_prompt)
        self.user_message_id = self.add_message("user", "")
        
        # Concise instructions
        default_instructions = """# WORKFLOW (Use specialized tools!)

1. **Find test** → `run_bash_cmd("grep -rn 'def test_name' tests/")`
2. **Read test** → `read_specific_lines("tests/file.py", START, END)`
3. **Find implementation** → `run_bash_cmd("grep -rn 'def function' --include='*.py' --exclude-dir=tests")`
4. **Read code** → `show_file("src/file.py")` or `read_specific_lines("src/file.py", START, END)`
5. **Run test** → `run_bash_cmd("python -m pytest test::name -xvs 2>&1 | tee /tmp/out.txt")`
6. **Understand error** → `run_bash_cmd("grep -E 'Error|Expected|Actual' /tmp/out.txt")`
7. **Make MINIMAL fix** → `insert_lines("file.py", LINE, "new code")` or `replace_in_file("file.py", FROM, TO, "new code")`
8. **Check syntax** → `check_syntax("file.py")`
9. **Review diff** → `run_bash_cmd("git diff file.py")`
10. **Test again** → `run_bash_cmd("python -m pytest test::name -xvs 2>&1 | tee /tmp/test.txt")`
11. **VERIFY "PASSED"** → `run_bash_cmd("grep PASSED /tmp/test.txt")` (CRITICAL!)
12. **Check regressions** → `run_bash_cmd("python -m pytest test_file.py -v")`
13. **If clean, finish** → `finish("Fixed X by doing Y")`

# CRITICAL RULES
- **Use specialized tools**: show_file, read_specific_lines, insert_lines, replace_in_file (safer than bash!)
- **Minimal changes**: Add 1-5 lines, NEVER rewrite functions
- **Verify "PASSED"**: grep must find it, not just exit code 0
- **Check regressions**: Full test file must pass
- **One tool per turn**: Observe before proceeding
- **Never modify tests**: They define correctness

# IF STUCK (>3 failures)
```
add_instructions_and_backtrack(
    instructions="New strategy: [describe]",
    at_message_id=15
)
```"""
        self.instructions_message_id = self.add_message("instructor", default_instructions)
        
        self.add_functions([self.finish])

    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the tree.

        The message must include fields: role, content, timestamp, unique_id, parent, children.
        Maintain a pointer to the current node and the root node.
        """
        message_id = len(self.id_to_message)
        message = {
            "role": role,
            "content": content,
            "timestamp": int(time.time()),
            "unique_id": message_id,
            "parent": self.current_message_id,
            "children": []
        }
        
        self.id_to_message.append(message)
        
        if self.current_message_id != -1:
            self.id_to_message[self.current_message_id]["children"].append(message_id)
        
        self.current_message_id = message_id
        
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
        path = []
        message_id = self.current_message_id
        
        while message_id != -1:
            path.append(message_id)
            message_id = self.id_to_message[message_id]["parent"]
        
        path.reverse()
        
        context_parts = []
        for msg_id in path:
            context_parts.append(self.message_id_to_context(msg_id))
        
        return "".join(context_parts)

    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        """
        for tool in tools:
            self.function_map[tool.__name__] = tool
    
    def finish(self, result: str):
        """Submit your solution. Generates git diff patch automatically.
        
        REQUIREMENTS BEFORE CALLING:
        1. All target tests MUST show "PASSED" in their output
        2. Full test file run without regressions
        3. Changes verified with git diff
        4. Syntax validated with python -m py_compile
        
        VERIFICATION CHECKLIST:
        ✓ Ran: python -m pytest path/to/test::test_name -xvs | grep "PASSED"
        ✓ Saw: "PASSED" in output (not just exit code 0)
        ✓ Ran: python -m pytest path/to/test_file.py -x  
        ✓ Saw: No FAILED or ERROR in full suite
        ✓ Ran: git diff (reviewed actual changes)
        
        Args:
            result (str): Summary of fix (patch auto-generated from git diff)
            
        Returns:
            Result summary (or raises error if requirements not met)
        """
        import subprocess
        
        try:
            enforce_comprehensive_testing(self.id_to_message)
        except ValueError as e:
            self.add_message("system", str(e))
            raise
        
        try:
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
            
            syntax_errors = []
            for file_path in modified_files:
                try:
                    import ast
                    with open(file_path, 'r') as f:
                        ast.parse(f.read(), filename=file_path)
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path}:{e.lineno}: {e.msg}")
                except FileNotFoundError:
                    pass
            
            if syntax_errors:
                error_msg = (
                    "❌ CANNOT FINISH: Python syntax errors detected!\n\n" +
                    "\n".join(syntax_errors) +
                    "\n\nFix these syntax errors before calling finish()."
                )
                raise ValueError(error_msg)
                
        except subprocess.TimeoutExpired:
            pass
        except ValueError:
            raise
        except Exception as e:
            print(f"Syntax validation warning: {e}")
        
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
            
            if not patch or len(patch.strip()) < 10:
                raise ValueError(
                    "❌ CANNOT FINISH: No changes detected!\n"
                    "You must ACTUALLY edit files before calling finish().\n"
                    "Use 'git diff' to verify you have made changes."
                )
            
            if patch.count('\n-') > patch.count('\n+') * 3:
                raise ValueError(
                    "⚠️ WARNING: Patch contains mostly deletions!\n"
                    "This might indicate file corruption. Review with 'git diff'."
                )
                
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"Patch validation warning: {e}")
        
        recent_messages = self.id_to_message[-20:]
        test_commands = [
            msg.get("content", "") 
            for msg in recent_messages 
            if msg.get("role") == "assistant"
        ]
        test_commands_str = " ".join(test_commands)
        
        has_run_tests = "run_tests" in test_commands_str or "pytest" in test_commands_str
        has_comprehensive_test = (
            "test_" in test_commands_str and 
            (".py" in test_commands_str or "tests/" in test_commands_str)
        )
        
        if not (has_run_tests and has_comprehensive_test):
            warning_msg = (
                "\n⚠️⚠️⚠️ WARNING: No evidence of comprehensive testing! ⚠️⚠️⚠️\n"
                "You SHOULD have run the ENTIRE test file/module to check for regressions.\n"
                "If you haven't done this, your changes might break existing tests (PASS_TO_PASS).\n"
                "Consider running: run_tests(test_path='tests/full_module_test.py')\n"
            )
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
        at_message_id = int(at_message_id)
        
        self.set_message_content(self.instructions_message_id, instructions)
        self.current_message_id = at_message_id
        
        self.failed_approaches.append(instructions[:50])
        self.consecutive_syntax_errors = 0
        self.consecutive_not_found = 0
        
        return f"Instructions updated and backtracked to message {at_message_id}."
    
    def _is_stuck(self, error_msg: str, function_name: str) -> bool:
        """
        Detect if agent is stuck in a failure loop.
        Returns True if automatic backtracking should be triggered.
        """
        if "SyntaxError" in error_msg or "IndentationError" in error_msg:
            self.consecutive_syntax_errors += 1
        else:
            self.consecutive_syntax_errors = 0
        
        if "FileNotFoundError" in error_msg or "No such file" in error_msg:
            self.consecutive_not_found += 1
        else:
            self.consecutive_not_found = 0
        
        action_key = f"{function_name}:{error_msg[:30]}"
        self.recent_actions.append(action_key)
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        
        stuck_patterns = [
            self.consecutive_syntax_errors >= 3,
            self.consecutive_not_found >= 3,
            len(self.recent_actions) >= 5 and len(set(self.recent_actions[-5:])) <= 2,
        ]
        
        return any(stuck_patterns)
    
    def _auto_backtrack(self, step: int):
        """
        Automatically backtrack with alternative strategies when stuck.
        """
        alternative_strategies = [
            "STUCK DETECTED! Try a different approach:\n"
            "1. Read MORE context (parent classes, related modules)\n"
            "2. Look for similar test cases that already pass\n"
            "3. Search for documentation or docstrings explaining expected behavior",
            
            "STILL STUCK! Different strategy:\n"
            "1. Use explore_codebase_deeply() to find all related code\n"
            "2. Look for inheritance hierarchies (base classes)\n"
            "3. Check if there's a different file that needs modification",
            
            "FINAL ATTEMPT! Radical rethink:\n"
            "1. Re-read the test requirement from scratch\n"
            "2. Maybe the fix location is COMPLETELY different\n"
            "3. Try find_failing_test() to understand what EXACTLY is being tested",
        ]
        
        strategy_index = len(self.failed_approaches) % len(alternative_strategies)
        strategy = alternative_strategies[strategy_index]
        
        backtrack_to = max(self.instructions_message_id + 1, self.current_message_id - 20)
        
        self.add_message("system", 
            f"🔄 AUTO-BACKTRACK TRIGGERED (Step {step})\n\n{strategy}\n\n"
            f"Backtracking to message {backtrack_to} to try fresh approach."
        )
        
        self.add_instructions_and_backtrack(strategy, backtrack_to)

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
        self.set_message_content(self.user_message_id, task)
        
        checkpoints = {
            10: "🎯 Checkpoint: Have you found the failing test yet?",
            20: "🎯 Checkpoint: Do you understand what the test expects?",
            35: "🎯 Checkpoint: Have you located the buggy code?",
            50: "🎯 Checkpoint: Have you tried a fix? Did tests pass?",
            70: "⚠️ Checkpoint: 70% through. Consider using add_instructions_and_backtrack if stuck.",
            85: "🚨 WARNING: Approaching step limit! Prioritize finishing over perfection.",
        }
        
        for step in range(max_steps):
            if step in checkpoints:
                self.add_message("system", f"\n{'='*60}\n{checkpoints[step]}\n{'='*60}\n")
            
            try:
                context = self.get_context()
                response = self.llm.generate(context)
                self.add_message("assistant", response)
                
                parsed = self.parser.parse(response)
                function_name = parsed["name"]
                arguments = parsed["arguments"]
                
                if function_name not in self.function_map:
                    error_msg = f"Error: Function '{function_name}' not found in available tools."
                    self.add_message("system", error_msg)
                    continue
                
                func = self.function_map[function_name]
                try:
                    result = func(**arguments)
                    
                    if function_name == "finish":
                        return result
                    
                    result_msg = f"Tool '{function_name}' executed successfully. Result:\n{result}"
                    self.add_message("system", result_msg)
                    
                except Exception as e:
                    error_msg = f"Error executing tool '{function_name}': {str(e)}"
                    self.add_message("system", error_msg)
                    
                    error_str = str(e)
                    if self._is_stuck(error_str, function_name):
                        self._auto_backtrack(step)
                        continue
                    
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
                error_str = str(e)
                error_msg = f"Error in agent loop: {error_str}"
                
                if "Missing ----BEGIN_FUNCTION_CALL----" in error_str:
                    error_msg += """

⚠️ RESPONSE FORMAT ERROR: You did not call any tool!

EVERY response MUST end with a function call. You cannot just write explanatory text.

WRONG:
  "I need to check the file. Should I proceed?"
  
CORRECT:
  "I need to check the file."
  ----BEGIN_FUNCTION_CALL----
  run_bash_cmd
  ----ARG----
  command
  cat path/to/file.py
  ----END_FUNCTION_CALL----

Choose one of these tools and call it NOW:
- run_bash_cmd(command="...") - Execute any bash command
- add_instructions_and_backtrack(instructions="...", at_message_id=N) - Try different approach
- finish(result="...") - Submit solution (only after tests pass)"""
                
                elif "Function name is empty" in error_str:
                    error_msg += """

⚠️ RESPONSE FORMAT ERROR: Function name is missing!

You wrote:
  ----BEGIN_FUNCTION_CALL--------END_FUNCTION_CALL----
  
This is incomplete. You must specify which function to call.

CORRECT format:
  ----BEGIN_FUNCTION_CALL----
  run_bash_cmd
  ----ARG----
  command
  ls -la
  ----END_FUNCTION_CALL----

Available functions: run_bash_cmd, add_instructions_and_backtrack, finish"""
                
                self.add_message("system", error_msg)
        
        return f"Agent reached maximum steps ({max_steps}) without completing the task."

    def message_id_to_context(self, message_id: int) -> str:
        """Convert a message id to a context string."""
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
    main()