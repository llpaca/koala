from google.genai import types
import subprocess


def read_file(path: str):
    with open(path, "r") as f:
        return f.read()


def write_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)

    return "File written successfully"


def run_command(command: str):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    return result.stdout + result.stderr


FUNCTION_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}

TOOLS = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "type": "function",
        "name": "run_command",
        "description": "Run shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
]