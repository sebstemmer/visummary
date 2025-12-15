from typing import NamedTuple, Callable


class SummarizeTranscriptsParams(NamedTuple):
    llm_model_id: str
    llm_system_and_user_prompt_to_response: Callable[[str, str], str]
    length_factor: int
    summary_format: str
