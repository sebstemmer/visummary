from typing import Callable
import ollama


def use(model: str) -> Callable[[str, str], str]:
    """
    Use a LLM that runs with Ollama.

    Args:
        model (str):
            The model ID to use e.g. "gemma_3__1b__q4km".

    Returns:
        Callable[[str, str], str]:
            A function that takes a system prompt and a user prompt as input and returns the response.
    """
    return lambda system_prompt, user_prompt: ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2, "top_p": 1.0, "repeat_penalty": 1.0},
    )["message"]["content"]
