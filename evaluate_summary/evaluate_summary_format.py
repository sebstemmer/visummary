import json
import os

import utils
from evaluate_summary import evaluate_summary_utils
from evaluate_summary.evaluate_summary_params import EvaluateSummaryParams
from summarize_transcripts import summarize_transcripts_utils


def evaluate(evaluate_summary_params: EvaluateSummaryParams):
    print(f"evaluate format of summary...")

    # handle already done

    format_evaluation_path = evaluate_summary_utils.get_format_evaluation_path(
        evaluate_summary_params=evaluate_summary_params
    )

    if os.path.isfile(format_evaluation_path):
        print(f"...format already evaluated")
        return

    # create system prompt

    num_sentences_placeholder_value = str(
        summarize_transcripts_utils.get_num_sentences(
            summarize_transcripts_params=evaluate_summary_params.summarize_transcripts_params,
        )
    )

    system_prompt_with_placeholders = """
        You are an AI judge that evaluates if a summary is in the correct format.
        
        YOUR TASK:
        
        You will receive a summary and you have to determine if it is in the required format.
        Give a short explanation (2 sentences) and the result of your analysis (True if its in the required format, False otherwise).
        
        REQUIRED FORMAT:
        
        {{summary_format}}
        
        OUTPUT FORMAT:
        
        Do only output the following JSON
        
        {
            "explanation": "string",
            "is_in_required_format: "boolean"
        }
    """

    system_prompt = system_prompt_with_placeholders.replace(
        "{{num_sentences}}", num_sentences_placeholder_value
    ).replace("{{summary_format}}", evaluate_summary_params.summary_format)

    # create user prompt

    user_prompt = summarize_transcripts_utils.get_summary(
        summarize_transcripts_params=evaluate_summary_params.summarize_transcripts_params,
    )

    # perform and save format evaluation as json

    format_evaluation_response = (
        evaluate_summary_params.llm_evaluator_system_and_user_prompt_to_response(
            system_prompt,
            user_prompt,
        )
    )

    format_evaluation_as_json = json.loads(format_evaluation_response)

    utils.save_json(
        path=format_evaluation_path,
        json_for_saving=format_evaluation_as_json,
    )

    print(f"...evaluated format of summary")
