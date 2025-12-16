import json
import os
from typing import List

import utils
from compare_summaries.compare_summaries_params import CompareSummariesParams
from summarize_transcripts.summarize_transcripts_params import (
    SummarizeTranscriptsParams,
)


def comparison_already_exists(compare_summaries_params: CompareSummariesParams) -> bool:
    return os.path.isfile(compare_summaries_params.comparison_path)


def get_comparison_pairs(
    compare_summaries_params: CompareSummariesParams,
) -> List[dict]:
    if not comparison_already_exists(compare_summaries_params=compare_summaries_params):
        return []

    with open(compare_summaries_params.comparison_path, "r", encoding="UTF-8") as f:
        return json.load(f)["pairs"]


def are_two_summaries_already_compared(
    compare_summaries_params: CompareSummariesParams,
    summary_params_a: SummarizeTranscriptsParams,
    summary_params_b: SummarizeTranscriptsParams,
) -> bool:
    pairs = get_comparison_pairs(compare_summaries_params=compare_summaries_params)

    for pair in pairs:
        if (
            pair["llm_model_id_a"] == summary_params_a.llm_model_id
            and pair["llm_model_id_b"] == summary_params_b.llm_model_id
        ):
            return True

    return False


def add_pair_to_comparison_pairs(
    compare_summaries_params: CompareSummariesParams, comparison_pair: dict
) -> None:
    if not comparison_already_exists(compare_summaries_params=compare_summaries_params):
        utils.save_json(
            path=compare_summaries_params.comparison_path,
            json_for_saving={"pairs": [comparison_pair]},
        )
        return

    pairs = get_comparison_pairs(compare_summaries_params=compare_summaries_params)
    pairs.append(comparison_pair)

    utils.save_json(
        path=compare_summaries_params.comparison_path,
        json_for_saving={"pairs": pairs},
    )
