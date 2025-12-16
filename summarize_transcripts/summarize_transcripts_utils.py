from chunk_audio import chunk_audio_utils
from summarize_transcripts.summarize_transcripts_params import (
    SummarizeTranscriptsParams,
)
from transcribe_audio_chunks import transcribe_audio_chunks_utils


def get_summaries_folder_path(
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> str:
    transcripts_folder_path = transcribe_audio_chunks_utils.get_transcripts_folder_path(
        transcribe_audio_chunks_params=summarize_transcripts_params.transcribe_audio_chunks_params,
    )
    return f"{transcripts_folder_path}/summaries"


def get_summary_folder_path(
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> str:
    summaries_folder_path = get_summaries_folder_path(
        summarize_transcripts_params=summarize_transcripts_params
    )
    return f"{summaries_folder_path}/{summarize_transcripts_params.llm_model_id}_{summarize_transcripts_params.length_factor}"


def get_summary_path(
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> str:
    summary_folder_path = get_summary_folder_path(
        summarize_transcripts_params=summarize_transcripts_params,
    )
    return f"{summary_folder_path}/summary.txt"


def get_summary(summarize_transcripts_params: SummarizeTranscriptsParams) -> str:
    summary_path = get_summary_path(
        summarize_transcripts_params=summarize_transcripts_params,
    )
    with open(summary_path, "r", encoding="UTF-8") as f:
        return f.read()


def get_num_sentences(
    summarize_transcripts_params: SummarizeTranscriptsParams,
) -> int:
    audio_length_in_ms = chunk_audio_utils.get_audio_length_in_ms(
        chunk_audio_params=summarize_transcripts_params.chunk_audio_params
    )

    return int(
        round(
            audio_length_in_ms
            / (1000.0 * 60.0 * summarize_transcripts_params.length_factor)
        )
    )
