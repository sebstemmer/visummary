from typing import Set

from common.utils import utils
from analysis.analyze_different_llms import (
    create_summary_for_video,
    evaluate_summary_format,
)
from analysis.analyze_different_llms.compare_summaries import compare_summaries

from common.local_ollama_llm import use_ollama_llm
from common.local_whispher_stt import transcribe_audio
from analysis.open_ai_api_llm import use_open_ai_api_llm

from faster_whisper import WhisperModel

from common.summarize_transcript import summarize_utils

reference_model_id = "gpt-5-mini"

reference_model = use_open_ai_api_llm.use(model="gpt-5-mini")

models = {
    "gemma_3__1b__q4km": use_ollama_llm.use(model="gemma3:1b-it-q4_K_M"),
    "llama_3_2__1b__q5km": use_ollama_llm.use(model="llama3.2:1b-instruct-q5_K_M"),
    "qwen_2_5__1_5b__q5km": use_ollama_llm.use(model="qwen2.5:1.5b-instruct-q5_K_M"),
    "qwen_2_5__3b__q5km": use_ollama_llm.use(model="qwen2.5:3b-instruct-q5_K_M"),
    "llama_3_2__3b__q5km": use_ollama_llm.use(model="llama3.2:3b-instruct-q5_K_M"),
    "gemma_3__4b__q4km": use_ollama_llm.use(model="gemma3:4b-it-q4_K_M"),
    "qwen_3__4b__q4km": use_ollama_llm.use(model="qwen3:4b-instruct-2507-q4_K_M"),
    "qwen_2_5__7b__q5km": use_ollama_llm.use(model="qwen2.5:7b-instruct-q5_K_M"),
    "deepseek_r1__8b": use_ollama_llm.use(model="deepseek-r1:8b"),
    "llama_3_1__8b__q5km": use_ollama_llm.use(model="llama3.1:8b-instruct-q5_K_M"),
    "gemma_3__12b__q4km": use_ollama_llm.use(model="gemma3:12b-it-q4_K_M"),
    "qwen_2_5__14b__q5km": use_ollama_llm.use(model="qwen2.5:14b-instruct-q5_K_M"),
    "gpt_oss__20b": use_ollama_llm.use(model="gpt-oss:20b"),
    "gemma_3__27b__q4km": use_ollama_llm.use(model="gemma3:27b-it-q4_K_M"),
    "qwen_3__30b__q4km": use_ollama_llm.use(model="qwen3:30b-a3b-instruct-2507-q4_K_M"),
    "deepseek_r1__32b": use_ollama_llm.use(model="deepseek-r1:32b"),
}

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

stt_model = WhisperModel(model_size_or_path="small", device="cpu", compute_type="int8")

stt_audio_path_to_transcript = lambda audio_path: transcribe_audio.transcribe(
    audio_path=audio_path, model=stt_model
)

length_factor = 3
base_path = "data/analyze_different_llms"

faithful_models_json_key = "faithful_models"
required_format_models_json_key = "required_format_models"


def create_video_base_path(video_id: str) -> str:
    return f"{base_path}/{video_id}"


def create_audio_chunks_data_path(video_base_path: str) -> str:
    return f"{video_base_path}/audio_chunks/data.json"


def create_transcript_data_path(video_base_path: str) -> str:
    return f"{video_base_path}/transcript/data.json"


def create_transcript_folder_path(video_base_path: str) -> str:
    return f"{video_base_path}/transcript"


def create_summary_as_json_path(video_base_path: str, llm_model_id: str) -> str:
    return f"{video_base_path}/llms/{llm_model_id}/summary_json.txt"


def create_summary_data_path(video_base_path: str, llm_model_id: str) -> str:
    return f"{video_base_path}/llms/{llm_model_id}/summary_data.json"


def create_format_evaluation_path(video_base_path: str, llm_model_id: str) -> str:
    return f"{video_base_path}/llms/{llm_model_id}/format_evaluation.json"


def create_comparison_path(video_base_path: str) -> str:
    return f"{video_base_path}/evaluation/comparison.json"


def create_comparison_points_path(video_base_path: str) -> str:
    return f"{video_base_path}/evaluation/comparison_points.json"


# create summaries

for video_id, video_url in videos.items():
    create_summary_for_video.create(
        video_id=video_id,
        create_video_base_path=create_video_base_path,
        video_url=video_url,
        create_audio_chunks_data_path=create_audio_chunks_data_path,
        create_transcript_data_path=create_transcript_data_path,
        stt_model=stt_model,
        llm_models=models,
        create_summary_as_json_path=create_summary_as_json_path,
        create_summary_data_path=create_summary_data_path,
        length_factor=length_factor,
        create_format_evaluation_path=create_format_evaluation_path,
    )

# only models that output with required format are considered

considered_models_set: Set[str] = set(models.keys())

for video_id in videos.keys():
    video_base_path = create_video_base_path(video_id)

    for llm_model_id in models.keys():
        format_evaluation_path = create_format_evaluation_path(
            video_base_path=video_base_path, llm_model_id=llm_model_id
        )

        is_in_required_format = evaluate_summary_format.is_in_required_format(
            format_evaluation_path
        )

        if (not is_in_required_format) and (llm_model_id in considered_models_set):
            considered_models_set.remove(llm_model_id)

# output models that are able to output with required format for all videos

print("----")
print("considered models via format:")
considered_models: list = list(considered_models_set)
considered_models.sort()
[print(model) for model in considered_models]

# compare summaries

for video_id, video_url in videos.items():
    compare_summaries.compare(
        video_id=video_id,
        create_video_base_path=create_video_base_path,
        create_comparison_path=create_comparison_path,
        considered_models=considered_models,
        create_summary_as_json_path=create_summary_as_json_path,
        create_audio_chunks_data_path=create_audio_chunks_data_path,
        create_transcript_data_path=create_transcript_data_path,
        llm_evaluator_system_and_user_prompt_to_response=reference_model,
        create_comparison_points_path=create_comparison_points_path,
    )

# final points

faithfulness_key = "faithfulness"
coverage_key = "coverage"

total_points: dict = {
    llm_model_id: {
        faithfulness_key: 0,
        coverage_key: 0,
    }
    for llm_model_id in considered_models
}

max_possible_points = (len(considered_models) - 1) * len(videos)

for video_id in videos.keys():
    video_base_path = create_video_base_path(video_id=video_id)

    comparison_points_path = create_comparison_points_path(
        video_base_path=video_base_path
    )

    points = utils.load_json(comparison_points_path)

    for model_id in considered_models:
        total_points[model_id][faithfulness_key] += (
            points[model_id]["faithfulness"] / max_possible_points
        )
        total_points[model_id][coverage_key] += (
            points[model_id]["coverage"] / max_possible_points
        )


utils.save_json(
    path=f"{base_path}/total_points.json",
    json_for_saving=total_points,
)

# coverage score

print("----")

coverage_score = [
    (key, round(value[coverage_key], 3))  # type: ignore
    for key, value in total_points.items()
]
coverage_score.sort(key=lambda x: x[1], reverse=True)

print("coverage score:")
[print(p) for p in coverage_score]

# faithfulness score

print("----")

faithfulness_score = [
    (key, round(value[faithfulness_key], 3))  # type: ignore
    for key, value in total_points.items()
]
faithfulness_score.sort(key=lambda x: x[1], reverse=True)

print("faithfulness score:")
[print(p) for p in faithfulness_score]

# combined score

print("---")

combined_coverage_faithfulness_score = [
    (key, round(0.7 * value["faithfulness"] + 0.3 * value["coverage"], 3))  # type: ignore
    for key, value in total_points.items()
]

combined_coverage_faithfulness_score.sort(key=lambda x: x[1], reverse=True)

print("combined coverage and faithfulness score:")
[print(p) for p in combined_coverage_faithfulness_score]

# speed factor

speed_relative_to_model_id = "qwen_3__30b__q4km"

speed_factor: dict = {llm_model_id: 0 for llm_model_id in considered_models}

for video_id in videos.keys():
    video_base_path = create_video_base_path(video_id=video_id)

    comparison_points_path = create_comparison_points_path(
        video_base_path=video_base_path
    )

    reference_summary_data_path = create_summary_data_path(
        video_base_path=video_base_path, llm_model_id=speed_relative_to_model_id
    )

    reference_summary_time_in_s = utils.load_json(path=reference_summary_data_path)[
        summarize_utils.summary_time_in_s_key
    ]

    for llm_model_id in considered_models:
        summary_data_path = create_summary_data_path(
            video_base_path=video_base_path, llm_model_id=llm_model_id
        )

        summary_time_in_s = summary_data = utils.load_json(path=summary_data_path)[
            summarize_utils.summary_time_in_s_key
        ]

        speed_factor[llm_model_id] += reference_summary_time_in_s / (
            summary_time_in_s * len(videos)
        )

print("----")

sorted_speed_factor = [(key, round(value, 3)) for key, value in speed_factor.items()]  # type: ignore
sorted_speed_factor.sort(key=lambda x: x[1], reverse=True)

print("speed factor:")
[print(s) for s in sorted_speed_factor]
