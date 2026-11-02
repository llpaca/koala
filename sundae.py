# this is sundae my local agent that is supposed to be usefull
from pathlib import Path
from llama_cpp import Llama

MODEL_PATH = "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODELS = {"int4": "qwen2.5-3b-instruct-q4_k_m.gguf"}

MODEL_DIR = Path.cwd() / "models"

CONTEXT_SIZE = 4096
BATCH_SIZE = 512
GPU_LAYERS = -1
MAX_TOKENS=4096

def LoadModel(model_path):
    print("[log] loading model..")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=CONTEXT_SIZE,
        n_gpu_layers=GPU_LAYERS,
        n_batch=BATCH_SIZE,
        flash_attn=True,
        type_k=8,type_v=8,
        verbose=False,
    )

    print("[log] Model ready\n")

    return llm
def generate_response(llm, messages):
    answer_parts = []

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repeat_penalty=1.05,
        stream=True,  # Enable live token streaming
    )

    print("\nSundae: ", end="", flush=True)

    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content")

        if token:
            answer_parts.append(token)
            print(token, end="", flush=True)

    answer = "".join(answer_parts)
    return answer
def chat(llm):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Give accurate, clear, and concise answers."
            ),
        }
    ]

    print("\nChat started. Commands: /exit, /quit, /clear")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in {"/exit", "/quit"}:
            print("Goodbye!")
            break

        if command == "/clear":
            messages = messages[:1]
            print("Conversation history cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            answer = generate_response(llm,messages,)
            messages.append({"role": "assistant", "content": answer})

        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            if messages and messages[-1]["role"] == "user":
                messages.pop()

        except Exception as exc:
            print(f"\nInference error: {exc}", file=sys.stderr)

            if messages and messages[-1]["role"] == "user":
                messages.pop()

def main():
    model_path = MODEL_PATH
    llm = LoadModel(model_path)
    chat(llm)

if __name__ == "__main__":
    main()