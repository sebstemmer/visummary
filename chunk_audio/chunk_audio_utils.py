import json

from chunk_audio.chunk_audio_params import ChunkAudioParams


def get_audio_chunks_folder_path(chunk_audio_params: ChunkAudioParams) -> str:
    return f"{chunk_audio_params.base_path}/audio_chunks_{chunk_audio_params.chunk_size_in_min}_{chunk_audio_params.overlap_in_percent}"


def get_audio_chunks_data_path(chunk_audio_params: ChunkAudioParams) -> str:
    audio_chunks_folder_path = get_audio_chunks_folder_path(
        chunk_audio_params=chunk_audio_params
    )
    return f"{audio_chunks_folder_path}/data.json"


def get_num_chunks(chunk_audio_params: ChunkAudioParams) -> int:
    audio_chunks_data_path = get_audio_chunks_data_path(
        chunk_audio_params=chunk_audio_params,
    )
    audio_chunks_data = json.load(
        open(
            audio_chunks_data_path,
        )
    )
    return audio_chunks_data["num_chunks"]


def get_audio_length_in_ms(chunk_audio_params: ChunkAudioParams) -> int:
    audio_chunks_data_path = get_audio_chunks_data_path(
        chunk_audio_params=chunk_audio_params,
    )
    audio_chunks_data = json.load(
        open(
            audio_chunks_data_path,
        )
    )
    return audio_chunks_data["audio_length_in_ms"]


def get_audio_chunk_path(audio_chunks_params: ChunkAudioParams, chunk_idx: int) -> str:
    audio_chunks_folder_path = get_audio_chunks_folder_path(
        chunk_audio_params=audio_chunks_params
    )
    return f"{audio_chunks_folder_path}/chunk_{chunk_idx}.mp3"
