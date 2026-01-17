from typing import Callable

from analysis.analyze_different_stts.compare_transcripts import compare_transcripts
from common.utils import utils
from analysis.analyze_different_stts.compare_transcripts.compare_transcripts import (
    calculate_points,
)
from analysis.analyze_different_stts.compare_transcripts.compare_transcripts_utils import (
    Transcript,
)
from common.chunk_audio import chunk_audio
from common.download_video import download_video
from common.local_whispher_stt import transcribe_audio
from analysis.open_ai_api_llm import use_open_ai_api_llm
from common.transcribe_audio_chunks import transcribe_audio_chunks
from faster_whisper import WhisperModel
import os


def analyze(
    video_id: str,
    create_video_base_path: Callable[[str], str],
    video_url: str,
    create_audio_chunks_data_path: Callable[[str], str],
    chunk_size_in_min: int,
    overlap_in_percent: int,
    create_transcript_data_path: Callable[[str, str], str],
    models: dict[str, WhisperModel],
    create_comparison_path: Callable[[str], str],
    reference_model_id: str,
    reference_model: WhisperModel,
    create_comparison_points_path: Callable[[str], str],
) -> None:
    """
    For a single video evaluates the transcripts produced by different STT models.

    It downloads the video, chunks it into audio chunks, transcribes the audio chunks using the
    reference STT model, and transcribes the audio chunks using the STT models that are evaluated.

    Then it compares all transcripts pairwise (1-vs-1) given the reference transcript as ground truth. The results of
    those comparisons are saved under comparison_path as JSON file. It calculates the points for each STT model based
    on the comparison results. These points are saved under comparison_points_path in another JSON file.

    The comparison is performed using a strong LLM that acts as an AI-as-a-judge.

    Args:
        video_id (str):
            Identifier of the video, whose audio is being transcribed.
        create_video_base_path (Callable[[str], str]):
            Function that takes a video id as input and creates a path to a folder containing all data related to
            transcribing the video and evaluating the transcripts.
        video_url (str):
            URL of the video.
        create_audio_chunks_data_path (Callable[[str], str]):
            Function that takes video_base_path as input and creates a path to the audio chunks data JSON file.
        chunk_size_in_min (int):
            Size of an audio chunk in minutes.
        overlap_in_percent (int):
            Percentage of overlap between consecutive chunks.
        create_transcript_data_path (Callable[[str, str], str]):
            Function that takes video_base_path as input and creates a path to the transcript data JSON file.
        models (dict[str, WhisperModel]):
            All STT models that are evaluated for the transcription. Maps the stt_model_id to the WhisperModel.
        create_comparison_path (Callable[[str], str]):
            Function that takes video_base_path as input and creates a path to the JSON file where all the comparison
            results are stored.
        reference_model_id (str):
            Id of the reference STT model which produces the reference transcript.
        reference_model (WhisperModel):
            Reference STT model.
        create_comparison_points_path (str):
            Function that takes video_base_path as input and creates a path to the JSON file where all the resulting
            comparison points are stored.

    Returns:
        None
    """
    print(f"analyzing {video_id}...")

    video_base_path = create_video_base_path(video_id)

    # download video

    audio_path = f"{video_base_path}/audio.webm"

    download_video.download(audio_path=audio_path, video_url=video_url)

    # chunk audio

    audio_chunks_data_path = create_audio_chunks_data_path(video_base_path)

    chunk_audio.chunk(
        audio_chunks_data_path=audio_chunks_data_path,
        audio_path=audio_path,
        chunk_size_in_min=chunk_size_in_min,
        overlap_in_percent=overlap_in_percent,
    )

    # create reference transcript

    reference_transcript_data_path = create_transcript_data_path(
        video_base_path, reference_model_id
    )

    transcribe_audio_chunks.transcribe(
        transcript_data_path=reference_transcript_data_path,
        audio_chunks_data_path=audio_chunks_data_path,
        stt_audio_path_to_transcript=lambda the_audio_path: transcribe_audio.transcribe(
            audio_path=the_audio_path, model=reference_model
        ),
    )

    # transcribe audio chunks

    for stt_model_id, stt_model in models.items():
        print(f"transcribe with model {stt_model_id}...")

        # transcribe

        transcript_data_path = create_transcript_data_path(
            video_base_path, stt_model_id
        )

        transcribe_audio_chunks.transcribe(
            transcript_data_path=transcript_data_path,
            audio_chunks_data_path=audio_chunks_data_path,
            stt_audio_path_to_transcript=lambda the_audio_path: transcribe_audio.transcribe(
                audio_path=the_audio_path, model=stt_model
            ),
        )

        print(f"...transcribed with model {stt_model_id}")

    # compare transcripts

    comparison_path = create_comparison_path(video_base_path)

    os.makedirs(os.path.dirname(comparison_path), exist_ok=True)

    transcripts = [
        Transcript(
            model_id=stt_model_id,
            transcript_data_path=create_transcript_data_path(
                video_base_path, stt_model_id
            ),
        )
        for stt_model_id in models.keys()
    ]

    compare_transcripts.compare(
        audio_chunks_data_path=audio_chunks_data_path,
        reference_transcript_data_path=reference_transcript_data_path,
        transcripts=transcripts,
        comparison_path=comparison_path,
        llm_evaluator_system_and_user_prompt_to_response=use_open_ai_api_llm.use(
            model="gpt-5-mini"
        ),
    )

    stt_model_id_to_points = calculate_points(
        comparison_path=comparison_path, transcripts=transcripts
    )

    comparison_points_path = create_comparison_points_path(video_base_path)

    utils.save_json(
        path=comparison_points_path,
        json_for_saving=stt_model_id_to_points,
    )

    print(f"...analyzed {video_id}")
