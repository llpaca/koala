from openai import OpenAI


def nvidia_nemo(
    client: OpenAI,
    memory: list,
    input: str = "",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    thinking: bool = True,
    tools: list | None = None,
    stream: bool = False,
):
    messages = list(memory)
    if input:
        messages.append({"role": "user", "content": input})

    kwargs = dict(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": thinking},
            "reasoning_budget": 16384,
        },
        stream=stream,
    )

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    return client.chat.completions.create(**kwargs)