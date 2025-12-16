from typing import List

from chunk_audio import chunk_audio_utils
from transcribe_audio_chunks.transcribe_audio_chunks_params import (
    TranscribeAudioChunksParams,
)


def get_transcripts_folder_path(
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
) -> str:
    audio_chunks_folder_path = chunk_audio_utils.get_audio_chunks_folder_path(
        chunk_audio_params=transcribe_audio_chunks_params.chunk_audio_params
    )
    return f"{audio_chunks_folder_path}/transcripts_{transcribe_audio_chunks_params.stt_model_id}"


def get_transcript_path(
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
    chunk_idx: int,
) -> str:
    transcripts_folder_path = get_transcripts_folder_path(
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
    )
    return f"{transcripts_folder_path}/chunk_{chunk_idx}.txt"


def get_transcripts(
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
) -> List[str]:
    num_chunks = chunk_audio_utils.get_num_chunks(
        chunk_audio_params=transcribe_audio_chunks_params.chunk_audio_params
    )

    transcripts = []
    for chunk_idx in range(num_chunks):
        transcript_path = get_transcript_path(
            transcribe_audio_chunks_params=transcribe_audio_chunks_params,
            chunk_idx=chunk_idx,
        )

        with open(transcript_path, "r") as f:
            transcript = f.read()

            transcripts.append(transcript)

    return transcripts
