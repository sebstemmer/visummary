from typing import NamedTuple, Callable


class TranscribeAudioChunksParams(NamedTuple):
    stt_model_id: str
    sst_audio_path_to_transcript: Callable[[str], str]
