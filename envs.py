from utils import get_sb_environment
import subprocess
import shlex

class LimitsExceeded(Exception):
    """Raised when the agent has reached its step limit."""


class SWEEnvironment:
    """
    Minimal interface to the SWEBench execution environment.

    Students may use their own wrapper. The environment must expose:
    - execute(command: str) -> str: Run a shell command and return stdout, or raise ValueError on failure
    """

    def __init__(self, instance: dict):
        self.env = get_sb_environment(instance)
     
    # -------------------- REQUIRED TOOLS --------------------
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        try:
            output = self.env.execute(command)
            # Handle dict return type from minisweagent
            if isinstance(output, dict):
                output = output.get("output", str(output))
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ValueError(output)
        except TimeoutError:
            raise ValueError("TimeoutError")
        return output
    
    def git_diff_summary(self, path: str = "") -> str:
        """Return a concise summary of current changes (status + diff)."""
        try:
            scope = shlex.quote(path) if path else ""
            status = self.env.execute("git status --porcelain")
            if isinstance(status, dict):
                status = status.get("output", str(status))
            diff = self.env.execute(f"git --no-pager diff --unified=0 {scope}")
            if isinstance(diff, dict):
                diff = diff.get("output", str(diff))
            return f"STATUS:\n{status}\n\nDIFF (u=0):\n{diff}"
        except Exception as e:
            return f"Error getting git diff summary: {e}"
    
    def generate_patch(self, result: str) -> str:
        """
        Generate a patch from the result (for SWE-Bench) with better error handling.
        """
        try:
            # IMPROVEMENT #1: Better diagnostics for patch generation failures
            # First verify files were actually changed
            status = self.env.execute("git status --porcelain")
            if isinstance(status, dict):
                status = status.get("output", str(status))
            
            if not status.strip():
                return f"{result}\n\nWARNING: No file changes detected by git. Agent may have failed to write files correctly."
            
            # Show what will be included in the patch
            self.env.execute("git add -A")
            patch_output = self.env.execute("git diff --cached")
            
            # Handle dict return type from minisweagent
            if isinstance(patch_output, dict):
                patch_output = patch_output.get("output", str(patch_output))
            
            if patch_output.strip():
                return patch_output
            
            # Fallback: try without staging
            unstaged_patch = self.env.execute("git diff HEAD")
            if isinstance(unstaged_patch, dict):
                unstaged_patch = unstaged_patch.get("output", str(unstaged_patch))
            
            if unstaged_patch.strip():
                return unstaged_patch
            
            return f"{result}\n\nNo changes detected to generate a patch."
        except Exception as e:
            return f"{result}\n\nError running git commands: {e}"
    
    # -------------------- TODO(student): add more functions here if you want --------------------
    def replace_in_file(self, file_path: str, from_line: int, to_line: int, content: str) -> str:
        """
        [Optional] Replace the content of the file from the given line to the given line with the given content
        """
        try:
            # VALIDATION 1: Prevent modifying test files (they should already be correct!)
            if '/test' in file_path or file_path.startswith('test'):
                raise ValueError(
                    f"⛔ BLOCKED: Cannot modify test file '{file_path}'!\n"
                    f"Test files define the requirements - they should NOT be changed.\n"
                    f"You must fix the SOURCE CODE to make the tests pass.\n"
                    f"\nCommon mistake: Modifying test_*.py files instead of the actual implementation.\n"
                    f"Action: Find the source file being tested and modify that instead."
                )
            
            # Convert line numbers to integers (in case they come as strings from parser)
            from_line = int(from_line)
            to_line = int(to_line)
            
            # Read the current file content
            read_output = self.env.execute(f"cat {file_path}")
            # Handle dict return type from minisweagent
            if isinstance(read_output, dict):
                read_output = read_output.get("output", str(read_output))
            lines = read_output.split('\n')
            
            # Validate line numbers (1-indexed)
            if from_line < 1 or to_line < 1:
                raise ValueError("Line numbers must be >= 1")
            if from_line > to_line:
                raise ValueError("from_line must be <= to_line")
            if to_line > len(lines):
                raise ValueError(f"to_line {to_line} exceeds file length {len(lines)}")
            
            # VALIDATION 2: Check for catastrophic edits (V3 FIX)
            # Extract old content that will be replaced
            old_content_lines = lines[from_line-1:to_line]
            old_content = '\n'.join(old_content_lines)
            
            # Import and call validation function
            from agent import validate_edit_safety
            validate_edit_safety(file_path, from_line, to_line, old_content, content)
            
            # Replace the lines (convert to 0-indexed)
            # Split content on newlines so multi-line replacements work correctly
            content_lines = content.split('\n')
            new_lines = lines[:from_line-1] + content_lines + lines[to_line:]
            new_content = '\n'.join(new_lines)
            
            # Write the modified content back to the file
            # FIXED: Use heredoc for safe writing (handles large files, Unicode, no length limit)
            temp_file = f"/tmp/swe_replace_{abs(hash(file_path))}.tmp"
            
            # Write to temp file using heredoc (most reliable method)
            write_cmd = f"cat > {shlex.quote(temp_file)} << 'EDIT_EOF'\n{new_content}\nEDIT_EOF"
            self.env.execute(write_cmd)
            
            # Verify temp file was written correctly (catch corruption early)
            verify_cmd = f"wc -l {shlex.quote(temp_file)}"
            verify_output = self.env.execute(verify_cmd)
            if isinstance(verify_output, dict):
                verify_output = verify_output.get("output", str(verify_output))
            
            temp_line_count = int(verify_output.strip().split()[0])
            expected_line_count = len(new_lines)
            
            if temp_line_count == 0 and expected_line_count > 0:
                raise ValueError(f"Temp file is empty! Expected {expected_line_count} lines. Write failed!")
            
            # Atomic move (preserves permissions, safer than direct write)
            self.env.execute(f"mv {shlex.quote(temp_file)} {shlex.quote(file_path)}")
            
            # Final verification to catch any corruption
            verify_final_cmd = f"wc -l {shlex.quote(file_path)}"
            verify_final = self.env.execute(verify_final_cmd)
            if isinstance(verify_final, dict):
                verify_final = verify_final.get("output", str(verify_final))
            
            final_line_count = int(verify_final.strip().split()[0])
            
            if final_line_count == 0 and expected_line_count > 0:
                # Emergency: file corrupted, restore from git
                raise ValueError(
                    f"CRITICAL: File is empty after write! Expected {expected_line_count} lines.\n"
                    f"Attempting to restore from git..."
                )
                try:
                    self.env.execute(f"git checkout -- {shlex.quote(file_path)}")
                except:
                    pass  # If restore fails, at least we raised the error
            
            return f"Successfully replaced lines {from_line}-{to_line} in {file_path} ({final_line_count} lines total)"
        except Exception as e:
            raise ValueError(f"Error replacing content in file: {str(e)}")
    
    def show_file(self, file_path: str) -> str:
        """
        [Optional]Show the content of the file
        """
        try:
            output = self.env.execute(f"cat {file_path}")
            # Handle dict return type from minisweagent
            if isinstance(output, dict):
                output = output.get("output", str(output))
            return output
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

    def search_in_file(self, file_path: str, pattern: str) -> str:
        """Search for a pattern in a file and return the matching lines."""
        # FIXED: Use shlex.quote to prevent shell injection
        return self.run_bash_cmd(f"grep {shlex.quote(pattern)} {shlex.quote(file_path)}")

    def list_functions(self, file_path: str) -> str:
        """List function and class definitions in a Python file."""
        return self.run_bash_cmd(f"grep -E '^\\s*(def|class)\\s+' {file_path}")

    def search_codebase(self, pattern: str) -> str:
        """Search for a pattern recursively in the codebase."""
        # FIXED: Use shlex.quote to prevent shell injection
        return self.run_bash_cmd(f"grep -r {shlex.quote(pattern)} .")

    def run_tests(self, test_path: str = "") -> str:
        """
        Run tests using pytest. If test_path is provided, runs specific tests.
        Otherwise, runs all tests.
        """
        cmd = "python -m pytest -xvs"
        if test_path:
            cmd += f" {test_path}"
        return self.run_bash_cmd(cmd)

    def search_and_replace(self, file_path: str, old_text: str, new_text: str) -> str:
        """
        Search for a string in a file and replace all occurrences with a new string.
        FIXED: Uses Python for reliability (handles all special characters correctly).
        """
        # VALIDATION: Prevent modifying test files
        if '/test' in file_path or file_path.startswith('test'):
            raise ValueError(
                f"⛔ BLOCKED: Cannot modify test file '{file_path}'!\n"
                f"Test files define the requirements - they should NOT be changed.\n"
                f"You must fix the SOURCE CODE to make the tests pass."
            )
        
        # Use Python's string.replace() which is more reliable than sed
        py_script = f"""
import sys
try:
    with open({repr(file_path)}, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count occurrences
    count = content.count({repr(old_text)})
    if count == 0:
        print(f"Warning: Pattern not found in {repr(file_path)}", file=sys.stderr)
        sys.exit(0)
    
    # Replace all occurrences
    new_content = content.replace({repr(old_text)}, {repr(new_text)})
    
    # Write back atomically
    import tempfile, os, shutil
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname({repr(file_path)}))
    try:
        os.write(fd, new_content.encode('utf-8'))
        os.close(fd)
        shutil.move(temp_path, {repr(file_path)})
    except:
        os.close(fd)
        os.unlink(temp_path)
        raise
    
    print(f"Replaced {{count}} occurrence(s) in {file_path}")
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        
        output = self.run_bash_cmd(f"python3 -c {shlex.quote(py_script)}")
        return f"Successfully replaced text in {file_path}. Output:\n{output}"

    def check_python_syntax(self, file_path: str) -> str:
        """Check the syntax of a Python file."""
        return self.run_bash_cmd(f"python -m py_compile {file_path}")
    
    def show_lines_with_numbers(self, file_path: str, from_line: int, to_line: int) -> str:
        """
        Show specific lines from a file with line numbers and visible indentation.
        This helps you see the exact indentation before editing.
        
        Args:
            file_path: Path to the file
            from_line: Starting line number (1-indexed)
            to_line: Ending line number (1-indexed)
        
        Returns:
            Lines with numbers and indentation markers
        """
        try:
            # Use sed to extract the lines and nl to add line numbers
            output = self.run_bash_cmd(f"sed -n '{from_line},{to_line}p' {file_path} | cat -A")
            # cat -A shows tabs as ^I and line ends as $, making indentation visible
            
            # Also show a cleaner version with line numbers
            clean_output = self.run_bash_cmd(f"sed -n '{from_line},{to_line}p' {file_path} | nl -ba -v {from_line}")
            
            return f"Lines {from_line}-{to_line} from {file_path}:\n\n{clean_output}\n\n(With indentation markers - spaces are normal, ^I means tab, $ means line end):\n{output}"
        except Exception as e:
            raise ValueError(f"Error reading lines from file: {str(e)}")
    
    # -------------------- ENHANCED TOOLS (V7) --------------------
    
    def show_file(self, file_path: str, max_lines: int = 500) -> str:
        """
        Show file contents with line numbers (easier than using cat -n or nl).
        
        Args:
            file_path: Path to file to display
            max_lines: Maximum number of lines to show (default 500, prevents huge outputs)
        
        Returns:
            File contents with line numbers
        """
        try:
            # Read file and add line numbers
            output = self.run_bash_cmd(f"nl -ba {shlex.quote(file_path)} | head -n {max_lines}")
            
            # Check if file was truncated
            line_count_output = self.run_bash_cmd(f"wc -l {shlex.quote(file_path)}")
            if isinstance(line_count_output, dict):
                line_count_output = line_count_output.get("output", str(line_count_output))
            
            total_lines = int(line_count_output.strip().split()[0])
            
            if total_lines > max_lines:
                output += f"\n\n... [File has {total_lines} total lines, showing first {max_lines}. Use read_lines_from_file() for specific sections]"
            
            return output
        except Exception as e:
            raise ValueError(f"Error showing file '{file_path}': {str(e)}")
    
    def read_specific_lines(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        Read specific lines from a file with line numbers (easier than sed -n).
        
        Args:
            file_path: Path to file
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
        
        Returns:
            Specified lines with line numbers
        """
        try:
            start_line = int(start_line)
            end_line = int(end_line)
            
            if start_line < 1 or end_line < 1:
                raise ValueError("Line numbers must be >= 1")
            if start_line > end_line:
                raise ValueError(f"start_line ({start_line}) must be <= end_line ({end_line})")
            
            # Use sed to extract lines, then nl to add line numbers
            output = self.run_bash_cmd(
                f"sed -n '{start_line},{end_line}p' {shlex.quote(file_path)} | nl -ba -v {start_line}"
            )
            
            return f"Lines {start_line}-{end_line} from {file_path}:\n\n{output}"
        except Exception as e:
            raise ValueError(f"Error reading lines {start_line}-{end_line} from '{file_path}': {str(e)}")
    
    def check_syntax(self, file_path: str) -> str:
        """
        Check Python file syntax (easier than running python -m py_compile).
        
        Args:
            file_path: Path to Python file
        
        Returns:
            "✓ Syntax OK" if valid, or error message if syntax errors found
        """
        try:
            # Check if it's a Python file
            if not file_path.endswith('.py'):
                return f"⚠️ Warning: {file_path} is not a .py file, skipping syntax check"
            
            # Try to compile the file
            result = self.run_bash_cmd(f"python -m py_compile {shlex.quote(file_path)} 2>&1")
            
            # If no output, it succeeded
            if not result.strip():
                return f"✓ Syntax OK: {file_path}"
            
            # Check if it's just a success message
            if "Compiling" in result or result.strip() == "":
                return f"✓ Syntax OK: {file_path}"
            
            # Otherwise it's probably an error
            return f"✗ Syntax Error in {file_path}:\n{result}"
            
        except Exception as e:
            # Extract syntax error details
            error_msg = str(e)
            if "SyntaxError" in error_msg or "IndentationError" in error_msg:
                return f"✗ Syntax Error in {file_path}:\n{error_msg}"
            else:
                raise ValueError(f"Error checking syntax of '{file_path}': {error_msg}")
    
    def insert_lines(self, file_path: str, after_line: int, content: str) -> str:
        """
        Insert new lines after a specific line number (simpler than sed insertion).
        
        Args:
            file_path: Path to file
            after_line: Line number to insert after (0 = insert at beginning)
            content: Content to insert (can be multiple lines)
        
        Returns:
            Success message
        """
        try:
            # Validation: Prevent modifying test files
            if '/test' in file_path or file_path.startswith('test'):
                raise ValueError(
                    f"⛔ BLOCKED: Cannot modify test file '{file_path}'!\n"
                    f"Test files define the requirements - they should NOT be changed."
                )
            
            after_line = int(after_line)
            
            # Read current file
            read_output = self.run_bash_cmd(f"cat {shlex.quote(file_path)}")
            if isinstance(read_output, dict):
                read_output = read_output.get("output", str(read_output))
            lines = read_output.split('\n')
            
            # Validate line number
            if after_line < 0 or after_line > len(lines):
                raise ValueError(f"after_line must be between 0 and {len(lines)}")
            
            # Insert content
            content_lines = content.split('\n')
            new_lines = lines[:after_line] + content_lines + lines[after_line:]
            new_content = '\n'.join(new_lines)
            
            # Write back
            temp_file = f"/tmp/swe_insert_{abs(hash(file_path))}.tmp"
            write_cmd = f"cat > {shlex.quote(temp_file)} << 'INSERT_EOF'\n{new_content}\nINSERT_EOF"
            self.run_bash_cmd(write_cmd)
            self.run_bash_cmd(f"mv {shlex.quote(temp_file)} {shlex.quote(file_path)}")
            
            num_lines_inserted = len(content_lines)
            return f"✓ Inserted {num_lines_inserted} line(s) after line {after_line} in {file_path}"
            
        except Exception as e:
            raise ValueError(f"Error inserting lines into '{file_path}': {str(e)}")
    
    # -------------------- IMPROVEMENT #3 & #8: Enhanced Exploration Tools --------------------
    
    def find_failing_test(self, issue_description: str) -> str:
        """
        IMPROVEMENT #8: Smart test discovery based on issue description.
        Helps quickly locate the relevant test file and test function.
        
        Args:
            issue_description: The problem statement or keywords from the issue
        
        Returns:
            Paths to likely test files and test functions
        """
        try:
            import re
            results = []
            
            # Extract likely test names from description (e.g., test_something)
            test_keywords = re.findall(r'test_\w+', issue_description.lower())
            
            # Also extract class/function names that might be tested
            code_keywords = re.findall(r'\b[A-Z]\w+\b', issue_description)
            
            all_keywords = test_keywords + [kw.lower() for kw in code_keywords]
            
            if not all_keywords:
                results.append("No obvious test keywords found in description.")
                results.append("Trying generic test file search...")
            
            # Search for test files containing these keywords
            for keyword in all_keywords[:5]:  # Limit to first 5 to avoid spam
                try:
                    cmd = f"find . -name '*.py' -path '*/test*' | xargs grep -l '{keyword}' 2>/dev/null | head -5"
                    output = self.run_bash_cmd(cmd)
                    if output.strip():
                        results.append(f"Tests mentioning '{keyword}':\n{output}")
                except:
                    pass
            
            if not results:
                # Fallback: list test directories
                try:
                    test_dirs = self.run_bash_cmd("find . -type d -name '*test*' | head -10")
                    results.append(f"Test directories in project:\n{test_dirs}")
                except:
                    pass
            
            return "\n".join(results) if results else "No test files found. Try: grep -rn 'def test_' ."
            
        except Exception as e:
            return f"Error searching for tests: {str(e)}"
    
    def explore_codebase_deeply(self, topic: str) -> str:
        """
        IMPROVEMENT #3: Comprehensive exploration for complex issues.
        Finds classes, functions, imports, tests related to a topic.
        Use when initial investigation doesn't reveal the issue.
        
        Args:
            topic: The class/function/concept to explore
        
        Returns:
            Comprehensive findings about the topic
        """
        try:
            steps = [
                (f"grep -rn 'class.*{topic}' --include='*.py' 2>/dev/null | head -10", "Class definitions"),
                (f"grep -rn 'def.*{topic}' --include='*.py' 2>/dev/null | head -10", "Function definitions"),
                (f"grep -rn 'import.*{topic}' --include='*.py' 2>/dev/null | head -10", "Import statements"),
                (f"find . -name '*{topic.lower()}*.py' 2>/dev/null | head -10", "Related files"),
                (f"find . -name '*test*.py' | xargs grep -l '{topic}' 2>/dev/null | head -5", "Test files"),
            ]
            
            results = [f"=== Deep exploration of '{topic}' ===\n"]
            
            for cmd, description in steps:
                try:
                    output = self.run_bash_cmd(cmd)
                    if output.strip():
                        results.append(f"\n--- {description} ---")
                        results.append(output)
                except:
                    results.append(f"\n--- {description} ---")
                    results.append("(No results)")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error exploring codebase: {str(e)}"

class DumbEnvironment:
    """
    Dumb environment that just executes the command
    """

    def execute(self, command: str) -> str:
        """
        Run the command in bash and return the output

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        result = subprocess.run(command, capture_output=True, shell=True, check=False)
        output = f"--STDOUT--\n{result.stdout.decode()}\n--STDERR--\n{result.stderr.decode()}"
        if result.returncode:
            raise ValueError(output)
        return output
    
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        return self.execute(command)


class MinimalSWEEnvironment:
    """
    Minimal, safe tool surface for SWEBench execution.

    Exposes only essential capabilities:
    - run_bash_cmd: execute shell commands
    - show_file: display files with line numbers
    - read_specific_lines: read file sections
    - replace_in_file: safe in-place edits with validation
    - run_tests: run pytest (optionally scoped)
    - check_syntax: validate Python syntax
    - generate_patch: produce git diff for submission
    """

    def __init__(self, instance: dict):
        self.env = get_sb_environment(instance)

    def run_bash_cmd(self, command: str, description: str = "") -> str:
        """Run a shell command and return its output (stderr included on failure)."""
        try:
            output = self.env.execute(command)
            if isinstance(output, dict):
                output = output.get("output", str(output))
            return output
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ValueError(output)
        except TimeoutError:
            raise ValueError("TimeoutError")

    def show_file(self, file_path: str) -> str:
        """Show file contents with line numbers."""
        try:
            return self.run_bash_cmd(f"nl -ba {shlex.quote(file_path)}")
        except Exception as e:
            raise ValueError(f"Error showing file '{file_path}': {str(e)}")

    # (removed) read_specific_lines

    def run_project_tests(self, test_path: str = "") -> str:
        """Run tests with environment-aware fallbacks.

        Strategy:
        - Try pytest first (scoped to test_path if provided)
        - If pytest is unavailable, try common project runners
          (tests/runtests.py) when present
        - Always capture output (stderr included) for the agent to analyze
        """
        try:
            scope = f" {test_path}" if test_path else ""
            # Always succeed and capture output even if pytest is missing
            output = self.env.execute(f"python -m pytest -q{scope} 2>&1 || true")
            if isinstance(output, dict):
                output = output.get("output", str(output))
            if "No module named pytest" not in output:
                return output
            # Fallback: project-local runner
            fallback = self.env.execute(
                "if [ -f tests/runtests.py ]; then python tests/runtests.py -q 2>&1 || true; "
                "elif [ -f runtests.py ]; then python runtests.py -q 2>&1 || true; "
                "else echo 'pytest_unavailable_and_no_runner'; fi"
            )
            if isinstance(fallback, dict):
                fallback = fallback.get("output", str(fallback))
            return fallback
        except Exception as e:
            return f"Error running tests: {e}"

    def run_last_failed_tests(self) -> str:
        """Re-run last failed tests using pytest cache if available.

        Falls back gracefully when pytest or cache is unavailable.
        """
        try:
            output = self.env.execute("pytest -q --last-failed 2>&1 || true")
            if isinstance(output, dict):
                output = output.get("output", str(output))
            return output
        except Exception as e:
            return f"Error running last failed tests: {e}"

    def extract_failed_nodeids(self, test_output: str) -> str:
        """Extract pytest nodeids from test output and suggest a rerun command.

        Returns a newline-separated list and a suggested pytest command.
        """
        try:
            import re as _re
            candidates = set()
            for line in test_output.splitlines():
                # Match formats like: path/to/test.py::TestClass::test_name FAILED
                m = _re.search(r"(\S+::\S+)(?=\s+FAILED|\s+ERROR|\s+XFAIL|$)", line)
                if m:
                    candidates.add(m.group(1))
                # Also match: FAILED path/to/test.py::test_name
                m2 = _re.search(r"FAILED\s+(\S+::\S+)", line)
                if m2:
                    candidates.add(m2.group(1))
            if not candidates:
                return "No failed nodeids found in output."
            nodeids = sorted(candidates)
            rerun = "pytest -q " + " ".join(nodeids)
            return "\n".join(nodeids) + f"\n\nSuggested rerun:\n{rerun}"
        except Exception as e:
            return f"Error extracting nodeids: {e}"

    def suggest_related_tests(self, symbol: str) -> str:
        """Suggest tests referencing a symbol (function/class/name) under tests/.

        Returns matching test file paths (limited).
        """
        try:
            cmd = (
                f"find tests -type f -name '*test*.py' 2>/dev/null | xargs grep -n "
                f"{shlex.quote(symbol)} 2>/dev/null | head -50"
            )
            output = self.env.execute(cmd)
            if isinstance(output, dict):
                output = output.get("output", str(output))
            return output if output.strip() else f"No tests reference '{symbol}'."
        except Exception as e:
            return f"Error suggesting related tests: {e}"

    def replace_in_file(self, file_path: str, from_line: int, to_line: int, content: str) -> str:
        """Safely replace lines [from_line, to_line] with content (multiline supported)."""
        # Prevent modifying tests
        if '/test' in file_path or file_path.startswith('test'):
            raise ValueError(
                f"⛔ BLOCKED: Cannot modify test file '{file_path}'!\n"
                f"Fix the implementation instead."
            )

        from_line = int(from_line)
        to_line = int(to_line)
        if from_line < 1 or to_line < 1:
            raise ValueError("Line numbers must be >= 1")
        if from_line > to_line:
            raise ValueError("from_line must be <= to_line")

        # Read entire file
        read_output = self.env.execute(f"cat {file_path}")
        if isinstance(read_output, dict):
            read_output = read_output.get("output", str(read_output))
        lines = read_output.split('\n')
        if to_line > len(lines):
            raise ValueError(f"to_line {to_line} exceeds file length {len(lines)}")

        # Validate we are not deleting function/class definitions accidentally
        old_content = '\n'.join(lines[from_line - 1:to_line])
        try:
            from agent import validate_edit_safety
            validate_edit_safety(file_path, from_line, to_line, old_content, content)
        except Exception as e:
            raise

        # No-op/duplicate edit guard: avoid proceeding if replacement is identical
        if content.split('\n') == lines[from_line - 1:to_line]:
            raise ValueError(
                "No-op edit: replacement content is identical to existing lines."
            )

        new_lines = lines[:from_line - 1] + content.split('\n') + lines[to_line:]
        new_content = '\n'.join(new_lines)

        # Atomic write via heredoc to temp file
        temp_file = f"/tmp/swe_replace_{abs(hash(file_path))}.tmp"
        write_cmd = f"cat > {shlex.quote(temp_file)} << 'EDIT_EOF'\n{new_content}\nEDIT_EOF"
        self.env.execute(write_cmd)
        self.env.execute(f"mv {shlex.quote(temp_file)} {shlex.quote(file_path)}")
        # Post-write verification: confirm file actually changed
        try:
            after = self.env.execute(f"sed -n '{from_line},{to_line}p' {shlex.quote(file_path)}")
            if isinstance(after, dict):
                after = after.get("output", str(after))
            if after.strip() == old_content.strip():
                raise ValueError("Edit appears to have had no effect (content unchanged).")
        except Exception:
            # If verification fails silently, still return success message
            # so the agent can inspect with git diff
            pass
        return f"Replaced lines {from_line}-{to_line} in {file_path}"

    # (removed) run_tests

    # (removed) search_tests

    # (removed) search_codebase

    # (removed) check_syntax

    def git_diff_summary(self, path: str = "") -> str:
        """Return a concise summary of current changes.

        Shows porcelain status and a unified diff for quick validation.
        """
        try:
            scope = shlex.quote(path) if path else ""
            status = self.env.execute("git status --porcelain")
            if isinstance(status, dict):
                status = status.get("output", str(status))
            diff = self.env.execute(f"git --no-pager diff --unified=0 {scope}")
            if isinstance(diff, dict):
                diff = diff.get("output", str(diff))
            return f"STATUS:\n{status}\n\nDIFF (u=0):\n{diff}"
        except Exception as e:
            return f"Error getting git diff summary: {e}"

    def symbol_exists(self, symbol: str, search_path: str = "") -> str:
        """Search for a function/class/identifier definition or usage in the codebase.

        Returns matching lines (limited) or a clear 'not found' message.
        """
        try:
            base = search_path if search_path else "."
            cmd = (
                f"git grep -n {{sym}} -- {shlex.quote(base)} 2>/dev/null | head -50"
            )
            # Use parameter expansion safely
            output = self.env.execute(cmd.format(sym=shlex.quote(symbol)))
            if isinstance(output, dict):
                output = output.get("output", str(output))
            return output if output.strip() else f"No occurrences of '{symbol}' found under {base}."
        except Exception as e:
            return f"Error searching for symbol: {e}"

    def generate_patch(self, result: str) -> str:
        """Generate a git diff patch for submission."""
        try:
            status = self.env.execute("git status --porcelain")
            if isinstance(status, dict):
                status = status.get("output", str(status))
            if not status.strip():
                return f"{result}\n\nWARNING: No file changes detected by git."
            self.env.execute("git add -A")
            patch_output = self.env.execute("git diff --cached")
            if isinstance(patch_output, dict):
                patch_output = patch_output.get("output", str(patch_output))
            if patch_output.strip():
                return patch_output
            unstaged_patch = self.env.execute("git diff HEAD")
            if isinstance(unstaged_patch, dict):
                unstaged_patch = unstaged_patch.get("output", str(unstaged_patch))
            if unstaged_patch.strip():
                return unstaged_patch
            return f"{result}\n\nNo changes detected to generate a patch."
        except Exception as e:
            return f"{result}\n\nError running git commands: {e}"