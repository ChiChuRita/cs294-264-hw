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
    
    def generate_patch(self, result: str) -> str:
        """
        Generate a patch from the result (for SWE-Bench)
        """
        try:
            patch_output = self.env.execute("git add -A && git diff --cached")
            # Handle dict return type from minisweagent
            if isinstance(patch_output, dict):
                patch_output = patch_output.get("output", str(patch_output))
            if patch_output.strip():
                return patch_output
            else:
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