import os
from time import perf_counter
from typing import Callable

from common.utils import utils
from common.chunk_audio import chunk_audio_utils
from common.transcribe_audio_chunks import transcribe_audio_chunks_utils


def transcribe(
    transcript_data_path: str,
    audio_chunks_data_path: str,
    stt_audio_path_to_transcript: Callable[[str], str],
) -> None:
    """
    Transcribes the audio chunks using an STT model and persists the transcript snippets in .txt format.

    Args:
        transcript_data_path (str):
            Path to the transcript data JSON file; the transcript snippets are stored in the same directory.
        audio_chunks_data_path (str):
            Path to the audio chunks data JSON file.
        stt_audio_path_to_transcript (Callable[[str], str]):
            A function that takes the path to an audio chunk as input and returns its transcript.

    Returns:
        None
    """

    print(f"transcribing audio chunks...")

    # check if already done

    if os.path.isfile(transcript_data_path):
        print(f"...audio chunks already transcribed")
        return

    # create folder for transcripts if it does not already exist

    transcripts_folder_path = os.path.dirname(transcript_data_path)

    os.makedirs(transcripts_folder_path, exist_ok=True)

    # transcribe audio chunks

    num_chunks = chunk_audio_utils.get_num_chunks(
        audio_chunks_data_path=audio_chunks_data_path,
    )

    total_transcription_time_in_s = 0

    for chunk_idx in range(num_chunks):  # type: ignore
        print(f"transcribing audio chunk {chunk_idx + 1}/{num_chunks}...")

        audio_path = chunk_audio_utils.get_audio_chunk_path(
            audio_chunks_data_path=audio_chunks_data_path, chunk_idx=chunk_idx
        )

        transcript_path = transcribe_audio_chunks_utils.get_transcript_snippet_path(
            transcript_data_path=transcript_data_path,
            chunk_idx=chunk_idx,
        )

        start_time = perf_counter()

        transcript = stt_audio_path_to_transcript(audio_path)

        transcription_time_in_s = perf_counter() - start_time

        total_transcription_time_in_s += transcription_time_in_s

        # save transcript as txt

        utils.save_file(path=transcript_path, content_for_saving=transcript)

        print(f"...transcribed audio chunk {chunk_idx + 1}/{num_chunks}")

    transcription_speed = chunk_audio_utils.get_audio_length_in_ms(
        audio_chunks_data_path
    ) / (1000.0 * total_transcription_time_in_s)

    utils.save_json(
        path=transcript_data_path,
        json_for_saving={
            transcribe_audio_chunks_utils.total_transcription_time_in_s_key: total_transcription_time_in_s,
            transcribe_audio_chunks_utils.transcription_speed_key: transcription_speed,
        },
    )

    print(f"...transcribed audio chunks")
