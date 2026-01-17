from typing import Callable

from analysis.analyze_different_llms import evaluate_summary_format
from common.chunk_audio import chunk_audio
from common.download_video import download_video
from common.local_whispher_stt import transcribe_audio
from common.summarize_transcript import summarize_transcript
from common.transcribe_audio_chunks import transcribe_audio_chunks
from faster_whisper import WhisperModel


def create(
    video_id: str,
    create_video_base_path: Callable[[str], str],
    video_url: str,
    create_audio_chunks_data_path: Callable[[str], str],
    create_transcript_data_path: Callable[[str], str],
    stt_model: WhisperModel,
    llm_models: dict[str, Callable[[str, str], str]],
    create_summary_as_json_path: Callable[[str, str], str],
    create_summary_data_path: Callable[[str, str], str],
    length_factor: int,
    create_format_evaluation_path: Callable[[str, str], str],
):
    """
    Executes the end-to-end pipeline from downloading the video to summarizing it. It also checks whether the summary
    output conforms to the required format.

    Args:
        video_id (str):
            Identifier of the video to be summarized.
        create_video_base_path (Callable[[str], str]):
            Function that takes a video id as input and creates a path to a folder containing all data related to summarizing the video and evaluating the summary.
        video_url (str):
            URL of the video.
        create_audio_chunks_data_path (Callable[[str], str]):
            Function that takes video_base_path as input and creates a path to the audio chunks data JSON file.
        create_transcript_data_path (Callable[[str], str]):
            Function that takes video_base_path as input and creates a path to the transcript data JSON file.
        stt_model (WhisperModel):
            The WhisperModel used for transcribing audio chunks.
        llm_models (dict[str, Callable[[str, str], str]]):
            All LLM models that are evaluated for the summarization. Maps the llm_model_id to a function that takes the system prompt and the user prompt as input and returns the LLM response.
        create_summary_as_json_path (Callable[[str, str], str]):
            Function that takes video_base_path and llm_model_id as input and creates a path to the summary JSON file.
        create_summary_data_path (Callable[[str, str], str]):
            Function that takes video_base_path and llm_model_id as input and creates a path to the summary data JSON file.
        length_factor (int):
            Determines the length of the summary.
        create_format_evaluation_path (Callable[[str, str], str]):
            Function that takes video_base_path and llm_model_id as input and creates a path to the evaluation of the summary format.

    Returns:
        None
    """

    print(f"create summary for video {video_id}...")

    video_base_path = create_video_base_path(video_id)

    # download video

    audio_path = f"{video_base_path}/audio.webm"

    download_video.download(audio_path=audio_path, video_url=video_url)

    # chunk audio

    audio_chunks_data_path = create_audio_chunks_data_path(video_base_path)

    chunk_audio.chunk(
        audio_chunks_data_path=audio_chunks_data_path,
        audio_path=audio_path,
        chunk_size_in_min=3,
        overlap_in_percent=5,
    )

    # transcribe audio chunks

    transcript_data_path = create_transcript_data_path(video_base_path)

    transcribe_audio_chunks.transcribe(
        transcript_data_path=transcript_data_path,
        audio_chunks_data_path=audio_chunks_data_path,
        stt_audio_path_to_transcript=lambda audio_path: transcribe_audio.transcribe(
            audio_path=audio_path, model=stt_model
        ),
    )

    # create summaries and evaluate their format

    for llm_model_id, use_model in llm_models.items():
        print(f"summarize with model {llm_model_id}, and evaluate format...")

        summary_as_json_path = create_summary_as_json_path(
            video_base_path, llm_model_id
        )

        summary_data_path = create_summary_data_path(video_base_path, llm_model_id)

        summarize_transcript.summarize(
            summary_as_json_path=summary_as_json_path,
            audio_chunks_data_path=audio_chunks_data_path,
            length_factor=length_factor,
            transcript_data_path=transcript_data_path,
            llm_system_and_user_prompt_to_response=use_model,
            summary_data_path=summary_data_path,
        )

        format_evaluation_path = create_format_evaluation_path(
            video_base_path, llm_model_id
        )

        evaluate_summary_format.evaluate(
            format_evaluation_path=format_evaluation_path,
            audio_chunks_data_path=audio_chunks_data_path,
            length_factor=length_factor,
            summary_as_json_path=summary_as_json_path,
        )

        print(f"...summarized, and evaluated format")

    print(f"...analyzed {video_id}")
