import os

from pydub import AudioSegment

import utils
from chunk_audio import chunk_audio_utils
from chunk_audio.chunk_audio_params import ChunkAudioParams
from download_video import download_video_utils


def chunk(chunk_audio_params: ChunkAudioParams) -> None:
    print(f"chunk audio...")

    # check if already done

    audio_chunks_data_path = chunk_audio_utils.get_audio_chunks_data_path(
        chunk_audio_params=chunk_audio_params
    )

    if os.path.isfile(audio_chunks_data_path):
        print(f"...audio already chunked")
        return

    # create folder for chunks if it does not already exist

    audio_chunks_folder_path = chunk_audio_utils.get_audio_chunks_folder_path(
        chunk_audio_params=chunk_audio_params,
    )

    os.makedirs(audio_chunks_folder_path, exist_ok=True)

    # load audio

    audio_path = download_video_utils.get_audio_path(chunk_audio_params.base_path)
    audio = AudioSegment.from_file(audio_path)

    # calc parameters for chunking

    audio_length_in_ms: int = len(audio)

    chunk_size_in_ms = chunk_audio_params.chunk_size_in_min * 60 * 1000

    snippets = range(0, audio_length_in_ms, chunk_size_in_ms)

    overlap_in_ms = chunk_size_in_ms * (chunk_audio_params.overlap_in_percent / 100.0)

    # chunk audio

    num_chunks = 0
    for idx, start_in_ms in enumerate(snippets):
        print(f"create audio chunk {idx + 1}/{len(snippets)}")

        start_minus_overlap_in_ms = max(start_in_ms - overlap_in_ms, 0)

        end_plus_overlap_in_ms = min(
            start_in_ms + chunk_size_in_ms + overlap_in_ms, audio_length_in_ms
        )

        chunk_in_ms = audio[start_minus_overlap_in_ms:end_plus_overlap_in_ms]

        chunk_in_ms.export(
            chunk_audio_utils.get_audio_chunk_path(
                audio_chunks_params=chunk_audio_params,
                chunk_idx=idx,
            ),
            format="mp3",
        )

        num_chunks += 1

    # save audio_chunks_data as json

    utils.save_json(
        path=audio_chunks_data_path,
        json_for_saving={
            "num_chunks": num_chunks,
            "audio_length_in_ms": audio_length_in_ms,
        },
    )

    print(f"...chunked audio")
