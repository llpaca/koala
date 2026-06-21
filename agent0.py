import os
from dotenv import load_dotenv
load_dotenv() 

from google import genai
from agents.ggl import goog
# from agents.groq import groq
# from agents.mstrl import mstrl
from asset.ascii import asciii


asciii()

ORANGE = "\033[38;2;255;165;0m"
RESET = "\033[0m"

AKL = [
    'MISTRAL_API_KEY_{i}',
    'GOOGLE_API_KEY_{i}',
    'GROQ_API_KEY'
]

MEM = []

client = genai.Client(api_key=os.getenv(AKL[1].format(i=2)))

while(True):
    # inp = input("agent0 $> ")
    inp = input(f"{ORANGE}agent0 $> {RESET}")
    stream = goog(client,
                    memory=MEM,
                    input=inp,
                    thinking_level="low",
                    stream=True,
                    temperature=0.7)
                    
    reply_text = ""
    
    for event in stream:
        if event.event_type == "step.delta" and event.delta.type == "text":
            print(event.delta.text, end="")
            reply_text += event.delta.text
    print()  # newline after the full reply

    MEM.append({"type":"user_input", "content":[{"type":"text", "text":inp}]})
    MEM.append({"type": "model_output", "content": [{"type": "text", "text": reply_text}]})

# way to access
# os.getenv(AKL[0].format(i=2))