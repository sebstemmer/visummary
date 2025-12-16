from evaluate_summary.evaluate_summary_params import EvaluateSummaryParams
from summarize_transcripts import summarize_transcripts_utils
import json


def get_format_evaluation_path(
    evaluate_summary_params: EvaluateSummaryParams,
) -> str:
    summary_folder_path = summarize_transcripts_utils.get_summary_folder_path(
        summarize_transcripts_params=evaluate_summary_params.summarize_transcripts_params,
    )
    return f"{summary_folder_path}/format_evaluation.json"


def get_faithfulness_evaluation_path(
    evaluate_summary_params: EvaluateSummaryParams,
) -> str:
    summary_folder_path = summarize_transcripts_utils.get_summary_folder_path(
        summarize_transcripts_params=evaluate_summary_params.summarize_transcripts_params,
    )
    return f"{summary_folder_path}/faithfulness_evaluation.json"


def is_in_required_format(evaluate_summary_params: EvaluateSummaryParams) -> bool:
    format_evaluation_path = get_format_evaluation_path(evaluate_summary_params)
    format_evaluation = json.load(open(format_evaluation_path))
    return format_evaluation["is_in_required_format"]


def is_faithful(evaluate_summary_params: EvaluateSummaryParams) -> bool:
    faithfulness_evaluation_path = get_faithfulness_evaluation_path(
        evaluate_summary_params
    )
    faithfulness_evaluation = json.load(open(faithfulness_evaluation_path))
    return faithfulness_evaluation["is_faithful"]
