import os
from typing import Callable
from time import perf_counter

from common.summarize_transcript.summarize_utils import get_summary_json_format
from common.utils import utils
from common.summarize_transcript import summarize_utils
from common.transcribe_audio_chunks import transcribe_audio_chunks_utils
import re


def summarize(
    summary_as_json_path: str,
    audio_chunks_data_path: str,
    length_factor: int,
    transcript_data_path: str,
    llm_system_and_user_prompt_to_response: Callable[[str, str], str],
    summary_data_path: str,
) -> None:
    """
    Summarizes the transcript snippets and saves the summary in JSON format.

    Args:
        summary_as_json_path (str):
            Path to the summary JSON file.
        audio_chunks_data_path (str):
            Path to the audio chunks data JSON file.
        length_factor (int):
            Determines the length of the summary.
        transcript_data_path (str):
            Path to the transcript data JSON file; the transcript snippets are stored in the same directory.
        llm_system_and_user_prompt_to_response (Callable[[str, str], str]):
            Function that maps the system and user prompts to the LLM response.
        summary_data_path (str):
            Path to the summary data JSON file.

    Returns:
        None
    """

    print(f"summarizing transcript...")

    # check if already done

    if os.path.isfile(summary_as_json_path):
        print(f"...transcripts already summarized")
        return

    # create folders if they dont exist yet

    os.makedirs(os.path.dirname(summary_as_json_path), exist_ok=True)

    # create system prompt

    num_sentences = summarize_utils.get_num_sentences(
        audio_chunks_data_path=audio_chunks_data_path, length_factor=length_factor
    )

    num_sentences_placeholder_value = str(num_sentences)

    system_prompt_with_placeholders = """
You are a transcript snippet summarizer.

YOU WILL RECEIVE:

* Consecutive transcript snippets from the same video.
* The snippets may overlap at their boundaries.

YOUR TASK:

* Determine what are the most important ideas, arguments, and conclusions from these snippets.
* Summarize them in exactly {{num_sentences}} medium-length sentences.
* Do not add information that does not appear in the transcript snippets.

OUTPUT FORMAT:

* Return ONLY valid JSON, no extra text.
* Return a summary that consists of exactly {{num_sentences}} medium-length sentences in the format
{{summary_format}}
"""

    system_prompt = system_prompt_with_placeholders.replace(
        "{{num_sentences}}", num_sentences_placeholder_value
    ).replace(
        "{{summary_format}}",
        get_summary_json_format(num_sentences),
    )

    # create user prompt

    transcripts = transcribe_audio_chunks_utils.get_transcript_snippets(
        transcript_data_path=transcript_data_path,
        audio_chunks_data_path=audio_chunks_data_path,
    )

    user_prompt_with_placeholders = (
        """TRANSCRIPT_SNIPPETS:

"""
        + "\n\n".join(
            [
                f"SNIPPET {transcript_idx}:\n\n{transcript}"
                for transcript_idx, transcript in enumerate(transcripts)
            ]
        )
        + """
----------------------------------

Remember to not add information that does not appear in the transcript snippets.
Remember to stick to the output format!

OUTPUT FORMAT:

* Return ONLY valid JSON, no extra text.
* Return a summary that consists of exactly {{num_sentences}} medium-length sentences in the format
{{summary_format}}
"""
    )

    user_prompt = user_prompt_with_placeholders.replace(
        "{{summary_format}}",
        get_summary_json_format(num_sentences),
    )

    # create summary via llm and save as txt, also track time if needed

    start_time = perf_counter()

    summary = llm_system_and_user_prompt_to_response(
        system_prompt,
        user_prompt,
    )

    summary_time_in_s = perf_counter() - start_time

    utils.save_json(
        path=summary_data_path,
        json_for_saving={summarize_utils.summary_time_in_s_key: summary_time_in_s},
    )

    cleaned_summary = _remove_code_fences(summary)

    utils.save_file(path=summary_as_json_path, content_for_saving=cleaned_summary)

    print("...summarized transcript")


def _remove_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
