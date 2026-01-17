from typing import Optional
from urllib.parse import parse_qs, urlparse

from faster_whisper import WhisperModel

from common.chunk_audio import chunk_audio
from common.download_video import download_video
from common.local_whispher_stt import transcribe_audio
from common.summarize_transcript import summarize_transcript, summarize_utils
from common.transcribe_audio_chunks import transcribe_audio_chunks
from common.local_ollama_llm import use_ollama_llm

whisper_model = "small"

stt_model = WhisperModel(
    model_size_or_path=whisper_model, device="cpu", compute_type="int8"
)

video_url = "https://www.youtube.com/watch?v=2MfQ2KCIUWo"

chunk_size_in_min = 3
overlap_in_percent = 5
length_factor = 3

llm_model = "qwen3:30b-a3b-instruct-2507-q4_K_M"

url = "https://www.youtube.com/watch?v=2MfQ2KCIUWo"

# create video_base_path

query = parse_qs(urlparse(url).query)
video_id: Optional[str] = query.get("v", [None])[0]  # type:ignore

if video_id is None:
    raise ValueError("video id not found in url")


video_base_path = f"data/main/{video_id}"

# download video

audio_path = f"{video_base_path}/audio.webm"

download_video.download(audio_path=audio_path, video_url=video_url)

# chunk audio

audio_chunks_data_path = f"{video_base_path}/audio_chunks/data.json"

chunk_audio.chunk(
    audio_chunks_data_path=audio_chunks_data_path,
    audio_path=audio_path,
    chunk_size_in_min=chunk_size_in_min,
    overlap_in_percent=overlap_in_percent,
)

# transcribe audio chunks

transcript_data_path = f"{video_base_path}/transcript/data.json"

transcribe_audio_chunks.transcribe(
    transcript_data_path=transcript_data_path,
    audio_chunks_data_path=audio_chunks_data_path,
    stt_audio_path_to_transcript=lambda audio_path: transcribe_audio.transcribe(
        audio_path=audio_path, model=stt_model
    ),
)

# summarize transcript

summary_as_json_path = f"{video_base_path}/summary/summary.json"
summary_data_path = f"{video_base_path}/summary/summary_data.json"

summarize_transcript.summarize(
    summary_as_json_path=summary_as_json_path,
    audio_chunks_data_path=audio_chunks_data_path,
    length_factor=length_factor,
    transcript_data_path=transcript_data_path,
    llm_system_and_user_prompt_to_response=use_ollama_llm.use(model=llm_model),
    summary_data_path=summary_data_path,
)

# output summary

print("----------------------------------\n")
summary = summarize_utils.get_summary(summary_as_json_path=summary_as_json_path)
print(summary)
