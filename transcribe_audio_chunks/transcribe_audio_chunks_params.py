from typing import Callable

from chunk_audio.chunk_audio_params import ChunkAudioParams


class TranscribeAudioChunksParams:
    def __init__(
        self,
        chunk_audio_params: ChunkAudioParams,
        stt_model_id: str,
        sst_audio_path_to_transcript: Callable[[str], str],
    ):
        self.base_path = chunk_audio_params.base_path
        self.chunk_audio_params = chunk_audio_params
        self.stt_model_id = stt_model_id
        self.sst_audio_path_to_transcript = sst_audio_path_to_transcript
