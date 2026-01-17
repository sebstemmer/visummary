from faster_whisper import WhisperModel


def transcribe(audio_path: str, model: WhisperModel) -> str:
    """
    Transcribe an audio file using a Whisper model.

    Args:
        audio_path (str):
            Path under which the audio file is stored.
        model (WhisperModel):
            Whisper model to use for transcription.

    Returns:
        str:
            Transcript.
    """
    segments, info = model.transcribe(audio_path)

    final_text = "".join([seg.text for seg in segments])

    return final_text
