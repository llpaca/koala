import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# for i in range(1,5):
#     my_var = os.getenv(f"GOOGLE_API_KEY_{i}")
#     print(my_var)

my_var = os.getenv(f"GOOGLE_API_KEY_3")

client = genai.Client(api_key=my_var)

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="write a c code to print using buffer and fprintf"

# )
# print(response.text)


# interaction = client.interactions.create(
#     model="gemini-3.5-flash",
#     # system_instruction="",
#     input="How does AI work?",
#     generation_config={
#         "thinking_level": "low",
#         # "temerature":1.0,
#     },
#     # stream=True
# )

# for event in stream:
#     if event.event_type == "step.delta":
#         if event.delta.type == "text":
#             print(event.delta.text, end="")

# print("Thoughts tokens:", interaction.usage.total_thought_tokens)
# print("Output tokens:", interaction.usage.total_output_tokens)

# print(interaction.output_text)

result = client.models.embed_content(
        model="gemini-embedding-2",
        contents="What is the meaning of life?",
        config=types.EmbedContentConfig(output_dimensionality=768)
)

[embedding_obj] = result.embeddings
embedding_length = len(embedding_obj.values)

print(f"Length of embedding: {embedding_length}")

print(result.embeddings)

# from google import genai

# client = genai.Client()

# uploaded_file = client.files.upload(file="path/to/organ.jpg")

# interaction = client.interactions.create(
#     model="gemini-3.5-flash",
#     input=[
#         {"type": "text", "text": "Tell me about this instrument"},
#         {
#             "type": "image",
#             "uri": uploaded_file.uri,
#             "mime_type": uploaded_file.mime_type
#         }
#     ]
# )
# print(interaction.output_text)