from common.utils import utils
from common.chunk_audio import chunk_audio_utils

summary_time_in_s_key = "summarize_time_in_s"


def get_num_sentences(audio_chunks_data_path: str, length_factor: int) -> int:
    """
    Get the number of required sentences for the summary.

    Args:
        audio_chunks_data_path (str):
            Path to the audio chunks data JSON file.
        length_factor (int):
            Determines the length of the summary.

    Returns:
        int:
            Number of required sentences.
    """
    audio_length_in_ms = chunk_audio_utils.get_audio_length_in_ms(
        audio_chunks_data_path=audio_chunks_data_path
    )

    return int(round(audio_length_in_ms / (1000.0 * 60.0 * length_factor)))


def get_summary(summary_as_json_path: str) -> str:
    """
    Convert the summary as JSON into the required text format.

    Args:
        summary_as_json_path (str):
            Path to the summary JSON file.

    Returns:
        str:
            Text format of the summary.
    """
    summary_as_json = utils.load_json(summary_as_json_path)

    bullets = "\n".join([f"* {sentence}" for sentence in summary_as_json.values()])

    result = f"""Key Insights:
    
{bullets}
"""

    return result


def get_summary_json_format(num_sentences: int) -> str:
    """
    Get the required summary JSON format to be used in LLM prompts.

    Args:
        num_sentences (str):
            Number of sentences that are required.

    Returns:
        str:
            JSON format of the summary.
    """
    bullets = ",\n    ".join(f'"{i+1}": String ({i+1}. sentence of summary)' for i in range(num_sentences))  # type: ignore

    return f"""{{
    {bullets}
}}
"""
