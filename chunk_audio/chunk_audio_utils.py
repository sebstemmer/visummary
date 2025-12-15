from chunk_audio.audio_chunks_params import AudioChunksParams
import json


def get_audio_chunks_folder_path(
    base_path: str, audio_chunks_params: AudioChunksParams
) -> str:
    return f"{base_path}/audio_chunks_{audio_chunks_params.chunk_size_in_min}_{audio_chunks_params.overlap_in_percent}"


def get_audio_chunks_data_path(audio_chunks_folder_path: str) -> str:
    return f"{audio_chunks_folder_path}/data.json"


def get_audio_chunk_path(audio_chunks_folder_path: str, chunk_idx: int) -> str:
    return f"{audio_chunks_folder_path}/chunk_{chunk_idx}.mp3"


def save_audio_chunks_data(
    audio_chunks_data_path: str,
    num_chunks: int,
    audio_length_in_ms: int,
) -> None:
    json.dump(
        {"num_chunks": num_chunks, "audio_length_in_ms": audio_length_in_ms},
        open(audio_chunks_data_path, "w"),
    )
