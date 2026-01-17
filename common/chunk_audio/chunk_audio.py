import os

from pydub import AudioSegment

from common.utils import utils
from common.chunk_audio import chunk_audio_utils


def chunk(
    audio_chunks_data_path: str,
    audio_path: str,
    chunk_size_in_min: int,
    overlap_in_percent: int,
) -> None:
    """
        Chunks the audio and saves the chunks in .mp3 format.

        Args:
            audio_chunks_data_path (str):
                Path to the audio chunks data JSON file; the audio chunks are created in the same directory.
            audio_path (str):
                Path to the .webm audio file.
            chunk_size_in_min (int):
                Size of an audio chunk in minutes.
            overlap_in_percent (int):
                Percentage of overlap between consecutive chunks.

    Returns:
        None

        Returns:
            None
    """
    print(f"chunk audio...")

    # check if already done

    if os.path.isfile(audio_chunks_data_path):
        print(f"...audio already chunked")
        return

    # create folders if they do not exist yet

    os.makedirs(os.path.dirname(audio_chunks_data_path), exist_ok=True)

    # load audio

    audio = AudioSegment.from_file(audio_path)

    # calc parameters for chunking

    audio_length_in_ms: int = len(audio)

    chunk_size_in_ms = chunk_size_in_min * 60 * 1000

    snippets = range(0, audio_length_in_ms, chunk_size_in_ms)  # type: ignore

    overlap_in_ms = chunk_size_in_ms * (overlap_in_percent / 100.0)

    # chunk audio

    num_chunks = 0
    for idx, start_in_ms in enumerate(snippets):
        print(f"create audio chunk {idx + 1}/{len(snippets)}...")

        start_minus_overlap_in_ms = max(start_in_ms - overlap_in_ms, 0)

        end_plus_overlap_in_ms = min(
            start_in_ms + chunk_size_in_ms + overlap_in_ms, audio_length_in_ms
        )

        chunk_in_ms = audio[start_minus_overlap_in_ms:end_plus_overlap_in_ms]

        chunk_in_ms.export(
            chunk_audio_utils.get_audio_chunk_path(
                audio_chunks_data_path=audio_chunks_data_path,
                chunk_idx=idx,
            ),
            format="mp3",
        )

        num_chunks += 1

        print(f"...audio chunk {idx + 1}/{len(snippets)} created")

    # save audio_chunks_data as json

    utils.save_json(
        path=audio_chunks_data_path,
        json_for_saving={
            chunk_audio_utils.num_chunks_key: num_chunks,
            chunk_audio_utils.audio_length_in_ms_key: audio_length_in_ms,
        },
    )

    print(f"...chunked audio")
