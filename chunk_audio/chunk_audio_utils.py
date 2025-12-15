from chunk_audio.audio_chunks_params import AudioChunksParams
import json


def get_audio_chunks_folder_path(
    base_path: str, audio_chunks_params: AudioChunksParams
) -> str:
    return f"{base_path}/audio_chunks_{audio_chunks_params.chunk_size_in_min}_{audio_chunks_params.overlap_in_percent}"


def get_audio_chunks_data_path(
    base_path: str, audio_chunks_params: AudioChunksParams
) -> str:
    audio_chunks_folder_path = get_audio_chunks_folder_path(
        base_path=base_path, audio_chunks_params=audio_chunks_params
    )
    return f"{audio_chunks_folder_path}/data.json"


def get_num_chunks(audio_chunks_data_path: str) -> int:
    audio_chunks_data = json.load(
        open(
            audio_chunks_data_path,
        )
    )
    return audio_chunks_data["num_chunks"]


def get_audio_chunk_path(
    base_path: str, audio_chunks_params: AudioChunksParams, chunk_idx: int
) -> str:
    audio_chunks_folder_path = get_audio_chunks_folder_path(
        base_path=base_path, audio_chunks_params=audio_chunks_params
    )
    return f"{audio_chunks_folder_path}/chunk_{chunk_idx}.mp3"
