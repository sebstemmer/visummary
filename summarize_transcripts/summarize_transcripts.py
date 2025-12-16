import os

from summarize_transcripts import summarize_transcripts_utils
from summarize_transcripts.summarize_transcripts_params import (
    SummarizeTranscriptsParams,
)
from transcribe_audio_chunks import transcribe_audio_chunks_utils


def summarize(
    summarize_transcripts_params: SummarizeTranscriptsParams,
):
    print(f"summarizing transcripts...")

    # check if already done

    summary_path = summarize_transcripts_utils.get_summary_path(
        summarize_transcripts_params=summarize_transcripts_params,
    )

    if os.path.isfile(summary_path):
        print(f"...transcripts already summarized")
        return

    # create folder for summary if it does not already exist

    summary_folder_path = summarize_transcripts_utils.get_summary_folder_path(
        summarize_transcripts_params=summarize_transcripts_params,
    )

    os.makedirs(summary_folder_path, exist_ok=True)

    # create system prompt

    num_sentences_placeholder_value = str(
        summarize_transcripts_utils.get_num_sentences(
            summarize_transcripts_params=summarize_transcripts_params,
        )
    )

    system_prompt_with_placeholders = """
        You will receive consecutive transcript snippets from the same video.
        Determine what are the {{num_sentences}} most important ideas, arguments, and conclusions from these snippets and summarize them in corresponding {{num_sentences}} middle-long sentences. 
        Do not add information that does not appear in the transcript snippets.
        Return the summary in the following format:
        
        {{summary_format}}
    """

    system_prompt = system_prompt_with_placeholders.replace(
        "{{num_sentences}}", num_sentences_placeholder_value
    ).replace("{{summary_format}}", summarize_transcripts_params.summary_format)

    # create user prompt

    transcripts = transcribe_audio_chunks_utils.get_transcripts(
        transcribe_audio_chunks_params=summarize_transcripts_params.transcribe_audio_chunks_params
    )

    user_prompt = "\n\n".join(
        [
            f"transcript snippet {transcript_idx}:\n\n{transcript}"
            for transcript_idx, transcript in enumerate(transcripts)
        ]
    )

    # create summary via llm and save as txt

    summary = summarize_transcripts_params.llm_system_and_user_prompt_to_response(
        system_prompt,
        user_prompt,
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print("...summarized transcripts")
