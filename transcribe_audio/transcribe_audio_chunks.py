import os

from chunk_audio import chunk_audio_utils
from chunk_audio.audio_chunks_params import AudioChunksParams
from transcribe_audio import transcribe_audio_chunks_utils
from transcribe_audio.transcribe_audio_chunks_params import TranscribeAudioChunksParams


def transcribe(
    base_path: str,
    audio_chunks_params: AudioChunksParams,
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
) -> None:
    audio_chunks_data_path = chunk_audio_utils.get_audio_chunks_data_path(
        base_path=base_path, audio_chunks_params=audio_chunks_params
    )

    print(f"transcribing audio chunks {audio_chunks_data_path}...")

    num_chunks = chunk_audio_utils.get_num_chunks(audio_chunks_data_path)

    transcripts_folder_path = transcribe_audio_chunks_utils.get_transcripts_folder_path(
        base_path=base_path,
        audio_chunks_params=audio_chunks_params,
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
    )

    if os.path.isdir(transcripts_folder_path):
        print(f"...audio chunks {audio_chunks_data_path} already transcribed")
        return

    os.makedirs(transcripts_folder_path)

    for chunk_idx in range(num_chunks):
        audio_path = chunk_audio_utils.get_audio_chunk_path(
            base_path=base_path,
            audio_chunks_params=audio_chunks_params,
            chunk_idx=chunk_idx,
        )
        transcript_path = transcribe_audio_chunks_utils.get_transcript_path(
            base_path=base_path,
            audio_chunks_params=audio_chunks_params,
            transcribe_audio_chunks_params=transcribe_audio_chunks_params,
            chunk_idx=chunk_idx,
        )

        transcript = transcribe_audio_chunks_params.sst_audio_path_to_transcript(
            audio_path
        )

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

    print(f"...transcribed {audio_chunks_data_path} audio chunks")
