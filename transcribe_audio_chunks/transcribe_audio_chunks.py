import os

from chunk_audio import chunk_audio_utils
from transcribe_audio_chunks import transcribe_audio_chunks_utils
from transcribe_audio_chunks.transcribe_audio_chunks_params import (
    TranscribeAudioChunksParams,
)


def transcribe(
    transcribe_audio_chunks_params: TranscribeAudioChunksParams,
) -> None:
    print(f"transcribing audio chunks...")

    # check if already done

    transcripts_folder_path = transcribe_audio_chunks_utils.get_transcripts_folder_path(
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
    )

    if os.path.isdir(transcripts_folder_path):
        print(f"...audio chunks already transcribed")
        return

    # create folder for transcripts if it does not already exist

    os.makedirs(transcripts_folder_path)

    # transcribe audio chunks

    num_chunks = chunk_audio_utils.get_num_chunks(
        chunk_audio_params=transcribe_audio_chunks_params.chunk_audio_params
    )

    for chunk_idx in range(num_chunks):
        print(f"transcribing audio chunk {chunk_idx + 1}/{num_chunks}")

        audio_path = chunk_audio_utils.get_audio_chunk_path(
            audio_chunks_params=transcribe_audio_chunks_params.chunk_audio_params,
            chunk_idx=chunk_idx,
        )
        transcript_path = transcribe_audio_chunks_utils.get_transcript_path(
            transcribe_audio_chunks_params=transcribe_audio_chunks_params,
            chunk_idx=chunk_idx,
        )

        transcript = transcribe_audio_chunks_params.sst_audio_path_to_transcript(
            audio_path
        )

        # save transcript as txt

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)

    print(f"...transcribed audio chunks")
