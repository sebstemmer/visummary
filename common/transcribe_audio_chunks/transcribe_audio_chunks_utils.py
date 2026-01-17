from typing import List

from common.utils import utils
from common.chunk_audio import chunk_audio_utils
import os


total_transcription_time_in_s_key = "total_transcription_time_in_s"
transcription_speed_key = "transcription_speed"


def get_transcription_speed(transcript_data_path: str) -> float:
    """
    Returns the unitless transcription speed (audio length devided by time to create the transcript), by loading it
    from the transcript data JSON file.

    Args:
        transcript_data_path (str):
            Path to the transcript data JSON file.

    Returns:
        float:
            Transcription speed.
    """
    transcripts_data = utils.load_json(transcript_data_path)
    return transcripts_data[transcription_speed_key]


def get_transcript_snippet_path(
    transcript_data_path: str,
    chunk_idx: int,
) -> str:
    """
    Get the path to a specific transcript snippet file.

    Args:
        transcript_data_path (str):
            Path to the transcript data JSON file.
        chunk_idx (int):
            Idx of the transcript snippet to receive the path.

    Returns:
        str:
            Path to the transcript snippet file.
    """
    transcript_folder_path = os.path.dirname(transcript_data_path)

    return f"{transcript_folder_path}/chunk_{chunk_idx}.txt"


def get_transcript_snippets(
    audio_chunks_data_path: str,
    transcript_data_path: str,
) -> List[str]:
    """
    Get all transcript snippets as list.

    Args:
        audio_chunks_data_path (str):
            Path to the audio chunks data JSON file.
        transcript_data_path (str):
            Path to the transcript data JSON file. The transcript snippets are stored in the same directory.

    Returns:
        List[str]:
            Each element in the list is a transcript snippet.
    """
    num_chunks = chunk_audio_utils.get_num_chunks(
        audio_chunks_data_path=audio_chunks_data_path
    )

    transcripts = []
    for chunk_idx in range(num_chunks):  # type: ignore
        transcript_path = get_transcript_snippet_path(
            transcript_data_path=transcript_data_path,
            chunk_idx=chunk_idx,
        )

        transcripts.append(utils.load_file(transcript_path))

    return transcripts
