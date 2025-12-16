from faster_whisper import WhisperModel

whisper_model = WhisperModel(
    model_size_or_path="tiny", device="cpu", compute_type="float32"
)


def transcribe(audio_path: str) -> str:
    segments, info = whisper_model.transcribe(audio_path)

    final_text = "".join([seg.text for seg in segments])

    return final_text
