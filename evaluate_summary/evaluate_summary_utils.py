import json

from chunk_audio.chunk_audio_params import ChunkAudioParams
from summarize_transcripts import summarize_transcripts_utils
from summarize_transcripts.summarize_transcripts_params import (
    SummarizeTranscriptsParams,
)
from transcribe_audio_chunks.transcribe_audio_chunks_params import (
    TranscribeAudioChunksParams,
)


def get_format_evaluation_path(
    base_path: str,
    chunk_audio_params: ChunkAudioParams,
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> str:
    summary_folder_path = summarize_transcripts_utils.get_summary_folder_path(
        base_path=base_path,
        chunk_audio_params=chunk_audio_params,
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
        summarize_transcripts_params=summarize_transcripts_params,
    )
    return f"{summary_folder_path}/format_evaluation.json"


def get_faithfulness_evaluation_path(
    base_path: str,
    chunk_audio_params: ChunkAudioParams,
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> str:
    summary_folder_path = summarize_transcripts_utils.get_summary_folder_path(
        base_path=base_path,
        chunk_audio_params=chunk_audio_params,
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
        summarize_transcripts_params=summarize_transcripts_params,
    )
    return f"{summary_folder_path}/faithfulness_evaluation.json"
