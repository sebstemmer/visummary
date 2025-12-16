from typing import Callable

from summarize_transcripts.summarize_transcripts_params import (
    SummarizeTranscriptsParams,
)


class EvaluateSummaryParams:
    def __init__(
        self,
        summarize_transcripts_params: SummarizeTranscriptsParams,
        llm_evaluator_system_and_user_prompt_to_response: Callable[[str, str], str],
    ):
        self.transcribe_audio_chunks_params = (
            summarize_transcripts_params.transcribe_audio_chunks_params
        )
        self.summarize_transcripts_params = summarize_transcripts_params
        self.summary_format = summarize_transcripts_params.summary_format
        self.llm_evaluator_system_and_user_prompt_to_response = (
            llm_evaluator_system_and_user_prompt_to_response
        )
