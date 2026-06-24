from google import genai
from google.genai import types
from .tools import TOOLS, FUNCTION_MAP

def goog(client, memory, input, thinking_level, stream, temperature):
        # client = genai.Client(api_key=user_api_key)
        turns = memory + [
                {"type":"user_input", "content":[{"type":"text", "text":input}]}
        ]
        return client.interactions.create(
            model="gemini-2.5-flash",
            # system_instruction="",
            input=turns,
            generation_config={
                "thinking_level": thinking_level,
                "temperature":temperature,
            },
            tools=TOOLS,
            stream=stream
        )