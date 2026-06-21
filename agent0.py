import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from agents.ggl import goog
from asset.ascii import asciii

from memory.manager import MemoryManager

asciii()
ORANGE = "\033[38;2;255;165;0m"
GREEN = "\033[92m"
RESET = "\033[0m"

AKL = [
    "MISTRAL_API_KEY_{i}",
    "GOOGLE_API_KEY_{i}",
    "GROQ_API_KEY",
]

client = genai.Client(api_key=os.getenv(AKL[1].format(i=2)))

memdb = MemoryManager()
MEM = []


def should_consider_memory(text: str) -> bool:
    text = text.strip()

    if len(text) < 20:
        return False

    junk = {
        "ok",
        "okay",
        "thanks",
        "thank you",
        "cool",
        "nice",
        "yep",
        "yes",
        "no",
        "hi",
        "hello"
    }

    if text.lower() in junk:
        return False

    return True

def retrieve_memories(query: str, k: int = 5) -> str:
    if len(memdb.store.documents) == 0:
        return ""

    try:
        results = memdb.store.search(query, k=k)
        memories = []

        for score, memory in results:
            if score > 0.65:
                memories.append(
                    f"[similarity={score:.2f}] {memory}"
                )

        return "\n".join(memories)

    except Exception as e:
        print(f"\nMemory Search Error: {e}")
        return ""


while True:
    try:
        inp = input(
            f"{ORANGE}agent0 $> {RESET}"
        ).strip()

        if not inp:
            continue

        if inp.lower() == "/memory":

            print("\n=== LONG TERM MEMORY ===")

            if not memdb.store.documents:
                print("No memories stored.")

            else:
                for i, memory in enumerate(
                    memdb.store.documents,
                    start=1
                ):
                    print(f"\n[{i}]")
                    print(memory)

            print(
                f"\nTotal Memories: "
                f"{len(memdb.store.documents)}\n"
            )

            continue

        if inp.lower() == "/memory count":

            print(
                f"Total Memories: "
                f"{len(memdb.store.documents)}"
            )

            continue


        if inp.lower().startswith("/memory search "):

            query = inp[len("/memory search "):]

            results = memdb.store.search(
                query,
                k=10
            )

            print(
                f"\n=== SEARCH RESULTS "
                f"FOR '{query}' ==="
            )

            for score, memory in results:

                print(
                    f"\nScore: {score:.4f}"
                )

                print(memory)

            print()

            continue


        if inp.lower() == "/memory clear":

            memdb.store.documents.clear()

            import faiss

            dim = (
                memdb.store.model
                .get_sentence_embedding_dimension()
            )

            memdb.store.index = faiss.IndexFlatIP(dim)

            memdb.save()

            print("Memory cleared.")

            continue

        if inp.lower() in {"exit", "quit"}:
            memdb.save()
            break

        memory_context = retrieve_memories(inp)

        enhanced_input = inp

        if memory_context:

            enhanced_input = f"""
                    Relevant memories from previous conversations:

                    {memory_context}

                    Current user message:

                    {inp}
                    """

        stream = goog(
            client,
            memory=MEM,
            input=enhanced_input,
            thinking_level="low",
            stream=True,
            temperature=0.7,
        )

        reply_text = ""

        for event in stream:

            if (
                event.event_type == "step.delta"
                and event.delta.type == "text"
            ):
                print(event.delta.text, end="")
                reply_text += event.delta.text

        print()

        MEM.append({
            "type": "user_input",
            "content": [
                {
                    "type": "text",
                    "text": inp
                }
            ]
        })

        MEM.append({
            "type": "model_output",
            "content": [
                {
                    "type": "text",
                    "text": reply_text
                }
            ]
        })

        if len(MEM) > 100:
            MEM = MEM[-100:]

        if should_consider_memory(inp):

            action = memdb.process(inp)

            print(
                f"{GREEN}[memory: {action}]{RESET}"
            )

            memdb.save()

    except KeyboardInterrupt:

        print("\nSaving memory...")

        memdb.save()

        break

    except Exception as e:

        print(f"\nError: {e}")