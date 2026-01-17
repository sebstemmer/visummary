import json
from itertools import combinations
from typing import List, Callable

from analysis.analysis_utils import analysis_utils
from analysis.analyze_different_stts.compare_transcripts import (
    compare_transcripts_utils,
)
from analysis.analyze_different_stts.compare_transcripts.compare_transcripts_utils import (
    Transcript,
)
from common.transcribe_audio_chunks import transcribe_audio_chunks_utils


def compare(
    audio_chunks_data_path: str,
    reference_transcript_data_path: str,
    transcripts: List[Transcript],
    comparison_path: str,
    llm_evaluator_system_and_user_prompt_to_response: Callable[[str, str], str],
) -> None:
    """
    Compares all the given transcripts created by different STT models pairwise (1-vs-1) using a strong evaluator LLM.
    The result is saved in a JSON file at comparison_path. A reference transcript is used as a ground truth in each
    comparison. Each pair is also evaluated in reverse order to reduce ordering effects. For each comparison, the LLM
    judge produces a short explanation and one of three outcomes: A wins, B wins, or draw. A win yields 1 point
    for the winning model, while a draw yields 0.5 points for each model. The results from both directions
    (A vs. B and B vs. A) are aggregated as follows: if both directions agree, that result is used; if one direction
    yields a win and the reverse yields a draw, the win is kept; if the directions contradict each other, the comparison is treated as a draw.

    Args:
        audio_chunks_data_path (str):
            Path to the audio chunks data JSON file; the audio chunks are created in the same directory.
        reference_transcript_data_path (str):
           Path to the transcript JSON file of the reference transcript.
        transcripts (List[Transcript]):
            Transcripts that are being compared pairwise.
        comparison_path (Callable[[str], str]):
            Path to the JSON file where all the comparison results are stored.
        llm_evaluator_system_and_user_prompt_to_response (Callable[[str, str], str]):
            Strong LLM model that acts as AI-as-a-judge that evaluates the summaries. System prompt and user prompt are passed as input and the LLM response is returned.

        Returns:
            None
    """
    print("compare transcripts...")

    # handle pairs

    reference_transcript_snippets = (
        transcribe_audio_chunks_utils.get_transcript_snippets(
            audio_chunks_data_path=audio_chunks_data_path,
            transcript_data_path=reference_transcript_data_path,
        )
    )

    pairs = list(combinations(transcripts, 2))

    print(f"num pairs to compare: {len(pairs)}")

    for pair in pairs:
        _compare_two_transcripts_and_there_reverse(
            audio_chunks_data_path=audio_chunks_data_path,
            transcript_a=pair[0],  # type: ignore
            transcript_b=pair[1],  # type: ignore
            comparison_path=comparison_path,
            reference_transcript_snippets=reference_transcript_snippets,
            llm_evaluator_system_and_user_prompt_to_response=llm_evaluator_system_and_user_prompt_to_response,
        )

    print("...compared transcripts")


def _compare_two_transcripts_and_there_reverse(
    audio_chunks_data_path: str,
    transcript_a: Transcript,
    transcript_b: Transcript,
    comparison_path: str,
    reference_transcript_snippets: List[str],
    llm_evaluator_system_and_user_prompt_to_response: Callable[[str, str], str],
) -> None:
    """
    Compares two transcripts pairwise (1-vs-1), using the reference transcript snippets as the source of truth.
    Evaluates both directions (A vs. B and B vs. A) and stores the result in a JSON file at comparison_path.

    Args:
        audio_chunks_data_path (str):
            Path to the audio chunks data JSON file.
        transcript_a (Transcript):
            Transcript A to be compared.
        transcript_b (Transcript):
            Transcript B to be compared.
        comparison_path (str):
            Path to the JSON file where all the comparison results are stored.
        reference_transcript_snippets (List[str]):
            Reference transcript consisting of several snippets that serves as the source of truth.
        llm_evaluator_system_and_user_prompt_to_response (Callable[[str, str], str]):
            Strong LLM model that acts as AI-as-a-judge that evaluates the transcripts. System prompt and user prompt
            are passed as input and the LLM response is returned.

        Returns:
            None
    """
    print(
        f"comparing transcripts {transcript_a.model_id} vs. {transcript_b.model_id}..."
    )

    # handle already compared

    if compare_transcripts_utils.are_two_transcripts_already_compared(
        comparison_path=comparison_path,
        transcripts_a=transcript_a,
        transcripts_b=transcript_b,
    ):
        print(f"...transcripts already compared")
        return

    # compare a <-> b and b <-> a

    (winner_a_b, explanation_a_b) = _compare_transcript_a_and_b(
        audio_chunks_data_path=audio_chunks_data_path,
        transcript_a=transcript_a,
        transcript_b=transcript_b,
        reference_transcript_snippets=reference_transcript_snippets,
        llm_evaluator_system_and_user_prompt_to_response=llm_evaluator_system_and_user_prompt_to_response,
    )

    (winner_b_a, explanation_b_a) = _compare_transcript_a_and_b(
        audio_chunks_data_path=audio_chunks_data_path,
        transcript_a=transcript_b,
        transcript_b=transcript_a,
        reference_transcript_snippets=reference_transcript_snippets,
        llm_evaluator_system_and_user_prompt_to_response=llm_evaluator_system_and_user_prompt_to_response,
    )

    comparison_result = {
        "model_id_a": transcript_a.model_id,
        "model_id_b": transcript_b.model_id,
        "a_b": {
            "winner": winner_a_b,
            "explanation": explanation_a_b,
        },
        "b_a": {
            "winner": winner_b_a,
            "explanation": explanation_b_a,
        },
        "is_commutative": analysis_utils.are_results_commutative(
            a_b=winner_a_b, b_a=winner_b_a
        ),
    }

    # save result in json

    compare_transcripts_utils.add_pair_to_comparison_pairs(
        comparison_path=comparison_path,
        comparison_pair=comparison_result,
    )

    print(f"...compared transcripts")


def _compare_transcript_a_and_b(
    audio_chunks_data_path: str,
    transcript_a: Transcript,
    transcript_b: Transcript,
    reference_transcript_snippets: List[str],
    llm_evaluator_system_and_user_prompt_to_response: Callable[[str, str], str],
) -> tuple[str, str]:
    """
    Compares two transcripts pairwise (1-vs-1), using the reference transcript snippets as
    the source of truth.

    Args:
        audio_chunks_data_path (str):
            Path to the audio chunks data JSON file.
        transcript_a (Transcript):
            Transcript A to be compared.
        transcript_b (Transcript):
            Transcript B to be compared.
        reference_transcript_snippets (List[str]):
            Reference transcript consisting of several snippets that serves as the source of truth.
        llm_evaluator_system_and_user_prompt_to_response (Callable[[str, str], str]):
            Strong LLM model that acts as AI-as-a-judge that evaluates the transcripts. System prompt and user prompt
            are passed as input and the LLM response is returned.

        Returns:
            tuple[str, str]:
                The result of the comparison (winner A, B, or draw) and a short explanation.
    """
    # create system prompt

    system_prompt = """
        You are an AI judge that compares transcripts.
        
        YOU WILL RECEIVE:
        
        * A reference transcript from a video.
        * Two candidate transcripts (A and B) from the same video, each created using a different speech-to-text model.
        * Each transcript consists of several consecutive transcript snippets that may overlap at the beginning and end.

        YOUR TASK:
        
        * Using the reference transcript as the source of truth, compare the two candidate transcripts A and B.
        * There are three possible outcomes: A is better than B, B is better than A, or it is a draw.
        * Your judgment must be symmetric: Swapping transcript A and transcript B must not change the result, except for swapping the labels A and B.
        * Errors in fill-words are less severe than errors in names, numbers, or technical terms.
        * Provide a short explanation for your decision.
        
        OUTPUT FORMAT:
        
        Output only the following JSON
        
        {
            "explanation": "string",
            "winner": "A" | "B" | "draw"
        }
    """

    print(system_prompt)

    # create user prompt

    user_prompt_with_placeholders = """REFERENCE TRANSCRIPT:
        
{{reference_transcript}}
        
----------------------------------
    
CANDIDATE TRANSCRIPT A:
        
{{candidate_transcript_a}}
        
----------------------------------
    
CANDIDATE TRANSCRIPT B:
        
{{candidate_transcript_b}}
"""

    reference_transcript_placeholder_value = (
        analysis_utils.create_transcript_from_snippets(
            transcript_snippets=reference_transcript_snippets
        )
    )

    candidate_transcript_snippets_a = (
        transcribe_audio_chunks_utils.get_transcript_snippets(
            audio_chunks_data_path=audio_chunks_data_path,
            transcript_data_path=transcript_a.transcript_data_path,
        )
    )

    candidate_transcript_a_placeholder_value = (
        analysis_utils.create_transcript_from_snippets(
            transcript_snippets=candidate_transcript_snippets_a
        )
    )

    candidate_transcript_snippets_b = (
        transcribe_audio_chunks_utils.get_transcript_snippets(
            audio_chunks_data_path=audio_chunks_data_path,
            transcript_data_path=transcript_b.transcript_data_path,
        )
    )

    candidate_transcript_b_placeholder_value = (
        analysis_utils.create_transcript_from_snippets(
            transcript_snippets=candidate_transcript_snippets_b
        )
    )

    user_prompt = (
        user_prompt_with_placeholders.replace(
            "{{reference_transcript}}", reference_transcript_placeholder_value
        )
        .replace("{{candidate_transcript_a}}", candidate_transcript_a_placeholder_value)
        .replace("{{candidate_transcript_b}}", candidate_transcript_b_placeholder_value)
    )

    print(user_prompt)

    # perform comparison

    response = llm_evaluator_system_and_user_prompt_to_response(
        system_prompt, user_prompt
    )

    response_as_json = json.loads(response)

    return response_as_json["winner"], response_as_json["explanation"]


def calculate_points(
    comparison_path: str,
    transcripts: List[Transcript],
) -> dict[str, int]:
    """
    Calculates the resulting points for each STT model based on the comparison results stored in comparison_path.

    Args:
        comparison_path (str):
            Path to the JSON file where all the comparison results are stored.
        transcripts (List[Transcript]):
            List of transcripts that have been compared.

        Returns:
            dict[str, int]:
                stt_model_id to points.
    """
    # init points to 0
    model_id_to_points: dict[str, int] = {}
    for transcript in transcripts:
        model_id_to_points[transcript.model_id] = 0

    pairs = compare_transcripts_utils.get_comparison_pairs(
        comparison_path=comparison_path
    )

    for pair in pairs:
        model_id_a = pair["model_id_a"]
        model_id_b = pair["model_id_b"]

        if pair["is_commutative"] >= 1:
            if pair["a_b"]["winner"] == "A":
                model_id_to_points[model_id_a] += 1
            elif pair["a_b"]["winner"] == "B":
                model_id_to_points[model_id_b] += 1
            else:
                model_id_to_points[model_id_a] += 0.5
                model_id_to_points[model_id_b] += 0.5
        else:
            model_id_to_points[model_id_a] += 0.5
            model_id_to_points[model_id_b] += 0.5

    return model_id_to_points
