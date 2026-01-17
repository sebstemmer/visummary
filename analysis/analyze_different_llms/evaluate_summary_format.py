import json
import os

from common.utils import utils
from common.summarize_transcript import summarize_utils


def evaluate(
    format_evaluation_path: str,
    audio_chunks_data_path: str,
    length_factor: int,
    summary_as_json_path: str,
) -> None:
    """
    Evaluates whether the summary output conforms to the required JSON format and saves the result in a JSON file.

    Args:
        format_evaluation_path (str):
            Path to JSON file where the format evaluation result is saved.
        audio_chunks_data_path (str):
            Path to the audio chunks data JSON file.
        length_factor (int):
            Determines the length of the summary.
        summary_as_json_path (str):
            Path to the summary JSON file.

    Returns:
        None
    """

    print(f"evaluate format of summary...")

    # handle already done

    os.makedirs(
        os.path.dirname(format_evaluation_path),
        exist_ok=True,
    )

    num_sentences = summarize_utils.get_num_sentences(
        audio_chunks_data_path=audio_chunks_data_path, length_factor=length_factor
    )

    summary_as_json = utils.load_file(summary_as_json_path)

    evaluation_result_as_json = _is_in_valid_format(
        summary_as_json=summary_as_json, num_sentences=num_sentences
    )

    utils.save_json(
        path=format_evaluation_path, json_for_saving=evaluation_result_as_json
    )

    print(f"...evaluated format of summary")


def _is_in_valid_format(summary_as_json: str, num_sentences: int) -> dict:
    """
    Checks if summary is in valid format. Returns bool and explanation.

    Args:
        summary_as_json (str):
            The summary in JSON format.
        num_sentences (int):
            The exact number of sentences the summary must have.

    Returns:
        dict:
            Boolean flag if it is in valid format and explanation.
    """
    try:
        summary_as_json: dict = json.loads(summary_as_json)
    except:
        return {
            is_in_required_format_key: False,
            "reason": "invalid json",
        }

    expected_idx = 0
    for idx_as_str, sentence in summary_as_json.items():
        expected_idx += 1

        if expected_idx > num_sentences:
            return {
                is_in_required_format_key: False,
                "reason": "more than expected sentences",
            }

        if idx_as_str != str(expected_idx):
            return {
                is_in_required_format_key: False,
                "reason": "idx breaks defined pattern",
            }

        if len(sentence) < 30:
            return {
                is_in_required_format_key: False,
                "reason": "sentences too short",
            }

    if expected_idx != num_sentences:
        return {
            is_in_required_format_key: False,
            "reason": "less than expected sentences",
        }

    return {
        is_in_required_format_key: True,
        "reason": "passed",
    }


is_in_required_format_key = "is_in_required_format"


def is_in_required_format(format_evaluation_path: str) -> bool:
    """
    Returns if summary is in valid format by reading the corresponding JSON file.

    Args:
        format_evaluation_path (str):
            Path to JSON file where the format evaluation result is saved.

    Returns:
        bool:
            True if summary is in valid format, False otherwise.
    """
    format_evaluation = utils.load_json(format_evaluation_path)
    return format_evaluation[is_in_required_format_key]
