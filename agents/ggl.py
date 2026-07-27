# ggl.py - Google Gemini agent wrapper

from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional, AsyncGenerator
import json

from tools import GEMINI_TOOLS, FUNCTION_MAP, ToolResult
from config import config


class GoogleAgent:
    """Wrapper for Google Gemini API with tool support."""

    def __init__(self, api_key: str = None, model: str = None, system_prompt: str = ""):
        self.client = genai.Client(api_key=api_key or config.google_api_key)
        self.model = model or config.agent.google_model
        self.system_prompt = system_prompt
        self.tools = GEMINI_TOOLS
        self.function_map = FUNCTION_MAP

    def _build_contents(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        """Convert message history to Gemini contents format."""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # System prompt handled separately in config
                continue

            if role == "tool":
                # Tool result
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_function_response(
                        name=msg.get("name", ""),
                        response={"result": msg.get("content", "")}
                    )]
                ))
            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant with tool calls
                parts = []
                if content:
                    parts.append(types.Part.from_text(text=content))
                for tc in msg["tool_calls"]:
                    parts.append(types.Part.from_function_call(
                        name=tc["function"]["name"],
                        args=json.loads(tc["function"]["arguments"] or "{}")
                    ))
                contents.append(types.Content(role="model", parts=parts))
            else:
                # User or assistant message
                gemini_role = "user" if role == "user" else "model"
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                ))

        return contents

    def _build_tool_config(self) -> types.ToolConfig:
        """Build tool configuration."""
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.AUTO
            )
        )

    def _build_generate_config(self, stream: bool = False) -> types.GenerateContentConfig:
        """Build generation config."""
        return types.GenerateContentConfig(
            model=self.model,
            system_instruction=self.system_prompt if self.system_prompt else None,
            tools=self.tools,
            tool_config=self._build_tool_config(),
            temperature=config.agent.temperature,
            max_output_tokens=config.agent.max_output_tokens,
            stream=stream,
        )

    def execute_tools(self, function_calls: List[Any]) -> List[Dict[str, Any]]:
        """Execute function calls and return results."""
        results = []
        for fc in function_calls:
            fn_name = fc.name
            fn_args = fc.args or {}

            fn = self.function_map.get(fn_name)
            if not fn:
                result = ToolResult(False, "", f"Unknown tool: {fn_name}")
            else:
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    result = ToolResult(False, "", str(e))

            results.append({
                "name": fn_name,
                "result": str(result)
            })

        return results

    def chat(self, messages: List[Dict[str, Any]], stream: bool = False) -> Any:
        """Send chat completion request."""
        contents = self._build_contents(messages)
        config = self._build_generate_config(stream=stream)

        if stream:
            return self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config
            )
        else:
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

    def stream_tool_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stream a turn and extract tool calls."""
        tool_calls = []
        full_text = ""

        for chunk in self.chat(messages, stream=True):
            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        full_text += part.text
                    if part.function_call:
                        tool_calls.append({
                            "name": part.function_call.name,
                            "arguments": json.dumps(dict(part.function_call.args))
                        })

        return full_text, tool_calls


def create_google_agent(system_prompt: str = "", api_key: str = None, model: str = None) -> GoogleAgent:
    """Factory function to create a GoogleAgent."""
    return GoogleAgent(
        api_key=api_key or config.google_api_key,
        model=model or config.agent.google_model,
        system_prompt=system_prompt
    )