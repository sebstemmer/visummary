from typing import Callable
import ollama


def use(model: str) -> Callable[[str, str], str]:
    return lambda system_prompt, user_prompt: ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )["message"]["content"]
