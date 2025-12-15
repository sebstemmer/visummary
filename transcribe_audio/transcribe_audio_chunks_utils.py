from chunk_audio import chunk_audio_utils
from chunk_audio.audio_chunks_params import AudioChunksParams
from transcribe_audio.transcribe_audio_chunks_params import TranscribeAudioChunksParams


def get_transcripts_folder_path(
    base_path: str,
    audio_chunks_params: AudioChunksParams,
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
) -> str:
    audio_chunks_folder_path = chunk_audio_utils.get_audio_chunks_folder_path(
        base_path=base_path, audio_chunks_params=audio_chunks_params
    )
    return f"{audio_chunks_folder_path}/transcripts_{transcribe_audio_chunks_params.stt_model_id}"


def get_transcript_path(
    base_path: str,
    audio_chunks_params: AudioChunksParams,
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
    chunk_idx: int,
) -> str:
    transcripts_folder_path = get_transcripts_folder_path(
        base_path=base_path,
        audio_chunks_params=audio_chunks_params,
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
    )
    return f"{transcripts_folder_path}/chunk_{chunk_idx}.txt"
