import os
from typing import List, NamedTuple

from common.utils import utils


class Transcript(NamedTuple):
    """
    Represents a transcript with associated model ID and the path to the stored transcript data JSON file.
    The transcript snippets are in the same folder.
    """

    model_id: str
    transcript_data_path: str


def get_comparison_pairs(
    comparison_path: str,
) -> List[dict]:
    """
    Get list of all comparison pairs. One pair is the comparison of transcript A and B and B and A.

    Args:
        comparison_path (str):
            Path to the JSON file where all the comparison results are stored.

    Returns:
        List[dict]:
            List of comparison pairs.

    """
    if not _comparison_already_exists(comparison_path=comparison_path):
        return []

    return utils.load_json(comparison_path)["pairs"]


def are_two_transcripts_already_compared(
    comparison_path: str,
    transcripts_a: Transcript,
    transcripts_b: Transcript,
) -> bool:
    """
    Check if the two transcripts A and B have already been compared.

    Args:
        comparison_path (str):
            Path to the JSON file where all the comparison results are stored.
        transcripts_a (Transcript):
            Transcript A to be compared.
        transcripts_b (Summary):
            Transcript B to be compared.

    Returns:
        bool:
            True if they have been already compared, False otherwise.
    """
    pairs = get_comparison_pairs(comparison_path=comparison_path)

    for pair in pairs:
        if (
            pair["model_id_a"] == transcripts_a.model_id
            and pair["model_id_b"] == transcripts_b.model_id
        ):
            return True

    return False


def add_pair_to_comparison_pairs(comparison_path: str, comparison_pair: dict) -> None:
    """
    Add a comparison pair to the list of comparison pairs.

    Args:
        comparison_path (str):
            Path to the JSON file where all the comparison results are stored.
        comparison_pair (dict):
            Comparison pair to be added.

    Returns:
        None
    """
    if not _comparison_already_exists(comparison_path=comparison_path):
        utils.save_json(
            path=comparison_path,
            json_for_saving={"pairs": [comparison_pair]},
        )
        return

    pairs = get_comparison_pairs(comparison_path=comparison_path)
    pairs.append(comparison_pair)

    utils.save_json(
        path=comparison_path,
        json_for_saving={"pairs": pairs},
    )


def _comparison_already_exists(comparison_path: str) -> bool:
    """
    Checks if JSON file under comparison_path exists.

    Args:
        comparison_path (str):
            Path to the JSON file where all the comparison results are stored.

    Returns:
        bool:
            True if it exists, False otherwise.
    """
    return os.path.isfile(comparison_path)
