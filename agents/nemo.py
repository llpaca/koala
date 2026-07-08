from openai import OpenAI

def nvidia_nemo(client, memory:string, input:string, top_p:float, stream:bool, temperature:float, thinking:bool):

  return client.chat.completions.create(
  #   model="nvidia/nemotron-3-super-120b-a12b",
    model="nvidia/nemotron-3-ultra",
    messages=[{"role":"user","content":""}],
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
    stream=True
  )

# for chunk in completion:
#   if not chunk.choices:
#     continue
#   reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
#   if reasoning:
#     print(reasoning, end="")
#   if chunk.choices[0].delta.content is not None:
#     print(chunk.choices[0].delta.content, end="")