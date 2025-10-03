from utils import get_sb_environment
import subprocess

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
            
            # Replace the lines (convert to 0-indexed)
            new_lines = lines[:from_line-1] + [content] + lines[to_line:]
            new_content = '\n'.join(new_lines)
            
            # Write the modified content back to the file
            # Use printf to avoid issues with special characters
            escaped_content = new_content.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
            self.env.execute(f'printf "%s" "{escaped_content}" > {file_path}')
            
            return f"Successfully replaced lines {from_line}-{to_line} in {file_path}"
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