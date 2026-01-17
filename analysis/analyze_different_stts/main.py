from faster_whisper import WhisperModel

from common.utils import utils
from analysis.analyze_different_stts import analyze_single_video
from common.transcribe_audio_chunks import transcribe_audio_chunks_utils

reference_model_id = "large_v3_float_32"

print("initialize stt models...")

reference_model = WhisperModel(
    model_size_or_path="large-v3", device="cpu", compute_type="float32"
)

models = {
    "tiny_int8": WhisperModel(
        model_size_or_path="tiny", device="cpu", compute_type="int8"
    ),
    "base_int8": WhisperModel(
        model_size_or_path="base", device="cpu", compute_type="int8"
    ),
    "small_int8": WhisperModel(
        model_size_or_path="small", device="cpu", compute_type="int8"
    ),
    "medium_int8": WhisperModel(
        model_size_or_path="medium", device="cpu", compute_type="int8"
    ),
    "large_v3_turbo_int8": WhisperModel(
        model_size_or_path="large-v3-turbo", device="cpu", compute_type="int8"
    ),
}

print("...initialized stt models")

videos = {
    "money_macro_jobs": "https://www.youtube.com/watch?v=2MfQ2KCIUWo",
    "hank_green_ai_water": "https://www.youtube.com/watch?v=H_c6MWk7PQc",
    "cs230_intro_ml": "https://www.youtube.com/watch?v=_NLHFoVNlbg",
    "sabine_datas_center_space": "https://www.youtube.com/watch?v=t8x09q1MjcM",
    "cloud_girl_ai_engineer": "https://www.youtube.com/watch?v=hJgbjDNsUYs",
    "ai_engineer_kernel": "https://www.youtube.com/watch?v=hJgbjDNsUYs",
    "pf_conservative_youth": "https://www.youtube.com/watch?v=uBQWJtuzx5o",
    "money_macro_megatrends": "https://www.youtube.com/watch?v=uBQWJtuzx5o",
    "pf_kast": "https://www.youtube.com/watch?v=nFLq-MV-ohY",
    "turc_transformers_cnns": "https://www.youtube.com/watch?v=KnCRTP11p5U",
}

chunk_size_in_min = 3
overlap_in_percent = 5

base_path = "data/analyze_different_stts_v2"


def create_video_base_path(video_id: str) -> str:
    return f"{base_path}/{video_id}"


def create_audio_chunks_data_path(video_base_path: str) -> str:
    return f"{video_base_path}/audio_chunks/data.json"


def create_transcript_data_path(video_base_path: str, stt_model_id: str) -> str:
    return f"{video_base_path}/stt_models/{stt_model_id}/transcripts/data.json"


def create_comparison_path(video_base_path: str) -> str:
    return f"{video_base_path}/evaluation/comparison.json"


def create_comparison_points_path(video_base_path: str) -> str:
    return f"{video_base_path}/evaluation/comparison_points.json"


# analyze videos

for video_id, video_url in videos.items():
    analyze_single_video.analyze(
        video_id=video_id,
        create_video_base_path=create_video_base_path,
        video_url=video_url,
        create_audio_chunks_data_path=create_audio_chunks_data_path,
        chunk_size_in_min=chunk_size_in_min,
        overlap_in_percent=overlap_in_percent,
        create_transcript_data_path=create_transcript_data_path,
        models=models,
        create_comparison_path=create_comparison_path,
        reference_model_id=reference_model_id,
        reference_model=reference_model,
        create_comparison_points_path=create_comparison_points_path,
    )


# final points

total_points: dict = {stt_model_id: 0 for stt_model_id in models.keys()}

max_possible_points = (len(models) - 1) * len(videos)

for video_id in videos.keys():
    video_base_path = create_video_base_path(video_id=video_id)

    comparison_points_path = create_comparison_points_path(
        video_base_path=video_base_path
    )

    points = utils.load_json(comparison_points_path)

    for model_id in models.keys():
        total_points[model_id] += points[model_id] / max_possible_points


utils.save_json(
    path=f"{base_path}/total_points.json",
    json_for_saving=total_points,
)

print("----------------------------------")

score = [(key, round(value, 3)) for key, value in total_points.items()]  # type: ignore
score.sort(key=lambda x: x[1], reverse=True)

print("quality score:")
[print(p) for p in score]


# speed factor

speed_factor: dict = {stt_model_id: 0 for stt_model_id in models.keys()}

for video_id in videos.keys():
    video_base_path = create_video_base_path(video_id=video_id)

    comparison_points_path = create_comparison_points_path(
        video_base_path=video_base_path
    )

    for stt_model_id in models.keys():
        transcript_data_path = create_transcript_data_path(
            video_base_path=video_base_path, stt_model_id=stt_model_id
        )

        speed_factor[
            stt_model_id
        ] += transcribe_audio_chunks_utils.get_transcription_speed(
            transcript_data_path
        ) / len(
            videos
        )

print("----------------------------------")

sorted_speed_factor = [(key, round(value, 3)) for key, value in speed_factor.items()]  # type: ignore
sorted_speed_factor.sort(key=lambda x: x[1], reverse=True)

print("speed factor:")
[print(s) for s in sorted_speed_factor]
