from typing import List

from chunk_audio import chunk_audio_utils, chunk_audio_params
from chunk_audio.chunk_audio_params import ChunkAudioParams
from summarize_transcripts.summarize_transcripts_params import (
    SummarizeTranscriptsParams,
)
from transcribe_audio_chunks import transcribe_audio_chunks_utils
from transcribe_audio_chunks.transcribe_audio_chunks_params import (
    TranscribeAudioChunksParams,
)


def get_summary_folder_path(
    base_path: str,
    chunk_audio_params: ChunkAudioParams,
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> str:
    transcripts_folder_path = transcribe_audio_chunks_utils.get_transcripts_folder_path(
        base_path=base_path,
        chunk_audio_params=chunk_audio_params,
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
    )
    return f"{transcripts_folder_path}/summaries/{summarize_transcripts_params.llm_model_id}_{summarize_transcripts_params.length_factor}"


def get_summary_path(
    base_path: str,
    chunk_audio_params: ChunkAudioParams,
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> str:
    summary_folder_path = get_summary_folder_path(
        base_path=base_path,
        chunk_audio_params=chunk_audio_params,
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
        summarize_transcripts_params=summarize_transcripts_params,
    )
    return f"{summary_folder_path}/summary.txt"


def get_summary(
    base_path: str,
    chunk_audio_params: ChunkAudioParams,
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> str:
    summary_path = get_summary_path(
        base_path=base_path,
        chunk_audio_params=chunk_audio_params,
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
        summarize_transcripts_params=summarize_transcripts_params,
    )
    with open(summary_path, "r", encoding="UTF-8") as f:
        return f.read()


def get_num_sentences(
    base_path: str,
    chunk_audio_params: ChunkAudioParams,
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> int:
    audio_length_in_ms = chunk_audio_utils.get_audio_length_in_ms(
        base_path=base_path, chunk_audio_params=chunk_audio_params
    )

    return int(
        round(
            audio_length_in_ms
            / (1000.0 * 60.0 * summarize_transcripts_params.length_factor)
        )
    )
