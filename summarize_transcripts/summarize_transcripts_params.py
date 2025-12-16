from typing import Callable

from transcribe_audio_chunks.transcribe_audio_chunks_params import (
    TranscribeAudioChunksParams,
)


class SummarizeTranscriptsParams:
    def __init__(
        self,
        transcribe_audio_chunks_params: TranscribeAudioChunksParams,
        llm_model_id: str,
        llm_system_and_user_prompt_to_response: Callable[[str], str],
        length_factor: int,
        summary_format: str,
    ):
        self.chunk_audio_params = transcribe_audio_chunks_params.chunk_audio_params
        self.transcribe_audio_chunks_params = transcribe_audio_chunks_params
        self.llm_model_id = llm_model_id
        self.llm_system_and_user_prompt_to_response = (
            llm_system_and_user_prompt_to_response
        )
        self.length_factor = length_factor
        self.summary_format = summary_format
