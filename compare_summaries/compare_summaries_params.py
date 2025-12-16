from typing import Callable, List

from summarize_transcripts.summarize_transcripts_params import (
    SummarizeTranscriptsParams,
)
from transcribe_audio_chunks.transcribe_audio_chunks_params import (
    TranscribeAudioChunksParams,
)


class CompareSummariesParams:
    def __init__(
        self,
        comparison_path: str,
        transcribe_audio_chunks_params: TranscribeAudioChunksParams,
        summaries: List[SummarizeTranscriptsParams],
        llm_evaluator_system_and_user_prompt_to_response: Callable[[str, str], str],
    ):
        self.transcribe_audio_chunks_params = transcribe_audio_chunks_params
        self.comparison_path = comparison_path
        self.summaries = summaries
        self.llm_evaluator_system_and_user_prompt_to_response = (
            llm_evaluator_system_and_user_prompt_to_response
        )
