from common.utils import utils
import os


num_chunks_key = "num_chunks"
audio_length_in_ms_key = "audio_length_in_ms"


def get_num_chunks(audio_chunks_data_path: str) -> int:
    """
        Get the number of audio chunks (read from the JSON file under the audio_chunks_data_path).

        Args:
            audio_chunks_data_path (str):
                Path to the audio chunks data JSON file.

    Returns:
        int:
            The number of audio chunks.
    """
    audio_chunks_data = utils.load_json(audio_chunks_data_path)
    return audio_chunks_data[num_chunks_key]


def get_audio_length_in_ms(audio_chunks_data_path: str) -> int:
    """
        Get the length of the complete audio in ms (read from the JSON file under the audio_chunks_data_path).

        Args:
            audio_chunks_data_path (str):
                Path to the audio chunks data JSON file.

    Returns:
        int:
            The length of the complete audio in ms.
    """
    audio_chunks_data = utils.load_json(audio_chunks_data_path)
    return audio_chunks_data[audio_length_in_ms_key]


def get_audio_chunk_path(audio_chunks_data_path: str, chunk_idx: int) -> str:
    """
        Get the path of a specific audio chunk file.

        Args:
            audio_chunks_data_path (str):
                Path to the audio chunks data JSON file.
            chunk_idx (int):
                Idx of the audio chunk to receive the path.

    Returns:
        str:
            Path to the audio chunk file.
    """
    audio_chunks_folder_path = os.path.dirname(audio_chunks_data_path)
    return f"{audio_chunks_folder_path}/chunk_{chunk_idx}.mp3"
