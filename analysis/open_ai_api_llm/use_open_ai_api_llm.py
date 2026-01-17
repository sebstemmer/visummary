from black.lines import Callable
from openai import OpenAI

with open("analysis/open_ai_api_llm/api_key.txt", "r", encoding="UTF-8") as f:
    openai_api_key = f.read()

openai_client = OpenAI(api_key=openai_api_key)


def use(model: str) -> Callable[[str, str], str]:
    """
    Use the OpenAI ChatGPT models via OpenAI API.

    Args:
        model (str):
            The model ID to use e.g. "gpt-5-mini".

    Returns:
        Callable[[str, str], str]:
            A function that takes a system prompt and a user prompt as input and returns the response.
    """
    return (
        lambda system_prompt, user_prompt: openai_client.chat.completions.create(
            model=model,
            messages=[  # type: ignore
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        .choices[0]  # type: ignore
        .message.content
    )
