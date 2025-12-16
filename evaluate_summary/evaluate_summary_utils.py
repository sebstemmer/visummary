from evaluate_summary.evaluate_summary_params import EvaluateSummaryParams
from summarize_transcripts import summarize_transcripts_utils


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
