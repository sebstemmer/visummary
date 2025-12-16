from typing import List
from chunk_audio import chunk_audio
from chunk_audio.chunk_audio_params import ChunkAudioParams
from compare_summaries.compare_summaries_params import CompareSummariesParams
from download_video import download_video
from download_video.download_video_params import DownloadVideoParams
from evaluate_summary import (
    evaluate_summary_format,
    evaluate_summary_faithfulness,
    evaluate_summary_utils,
)
from evaluate_summary.evaluate_summary_params import EvaluateSummaryParams
from local_ollama_llm import use_ollama_llm
from open_ai_api_llm import use_open_ai_api_llm
from local_whispher_stt import transcribe_audio
from summarize_transcripts.summarize_transcripts_params import (
    SummarizeTranscriptsParams,
)
from summarize_transcripts import summarize_transcripts
from transcribe_audio_chunks import transcribe_audio_chunks
from transcribe_audio_chunks.transcribe_audio_chunks_params import (
    TranscribeAudioChunksParams,
)
from compare_summaries import compare_summaries

use_llm_model_key = "model"
gpt_5_mini_model_key = "gpt-5-mini"

models = {
    "llama_3_2__1b__q5km": {
        use_llm_model_key: use_ollama_llm.use(model="llama3.2:1b-instruct-q5_K_M")
    },
    "qwen_2_5__1_5b__q5km": {
        use_llm_model_key: use_ollama_llm.use(model="qwen2.5:1.5b-instruct-q5_K_M")
    },
    "qwen_2_5__3b__q5km": {
        use_llm_model_key: use_ollama_llm.use(model="qwen2.5:3b-instruct-q5_K_M")
    },
    "llama_3_2__3b__q5km": {
        use_llm_model_key: use_ollama_llm.use(model="llama3.2:3b-instruct-q5_K_M")
    },
    "qwen_3__4b__q4km": {
        use_llm_model_key: use_ollama_llm.use(model="qwen3:4b-instruct-2507-q4_K_M")
    },
    "llama_2__7b__q5km": {
        use_llm_model_key: use_ollama_llm.use(model="llama2:7b-chat-q5_K_M")
    },
    "qwen_2_5__7b__q5km": {
        use_llm_model_key: use_ollama_llm.use(model="qwen2.5:7b-instruct-q5_K_M")
    },
    "deepseek_r1__8b": {use_llm_model_key: use_ollama_llm.use(model="deepseek-r1:8b")},
    "llama_3_1__8b__q5km": {
        use_llm_model_key: use_ollama_llm.use(model="llama3.1:8b-instruct-q5_K_M")
    },
    "qwen_2_5__14b__q5km": {
        use_llm_model_key: use_ollama_llm.use(model="qwen2.5:14b-instruct-q5_K_M")
    },
    "gpt_oss__20b": {use_llm_model_key: use_ollama_llm.use(model="gpt-oss:20b")},
    gpt_5_mini_model_key: {
        use_llm_model_key: use_open_ai_api_llm.use(model="gpt-5-mini")
    },
    "qwen_3__30b__q4km": {
        use_llm_model_key: use_ollama_llm.use(
            model="qwen3:30b-a3b-instruct-2507-q4_K_M"
        )
    },
    "deepseek_r1__32b": {
        use_llm_model_key: use_ollama_llm.use(model="deepseek-r1:32b")
    },
}

# create audio transcripts

base_path = "data/money-macro-jobs"

download_video_params = DownloadVideoParams(
    base_path=base_path, video_url="https://www.youtube.com/watch?v=2MfQ2KCIUWo"
)

download_video.download(download_video_params=download_video_params)

chunk_audio_params = ChunkAudioParams(
    download_video_params=download_video_params,
    chunk_size_in_min=3,
    overlap_in_percent=5,
)

chunk_audio.chunk(chunk_audio_params=chunk_audio_params)

transcribe_audio_chunks_params = TranscribeAudioChunksParams(
    chunk_audio_params=chunk_audio_params,
    stt_model_id="local_tiny_whisper",
    stt_audio_path_to_transcript=lambda audio_path: transcribe_audio.transcribe(
        audio_path=audio_path
    ),
)

transcribe_audio_chunks.transcribe(
    transcribe_audio_chunks_params=transcribe_audio_chunks_params
)

# create summaries and evaluate format

summarize_transcripts_params_list = [
    SummarizeTranscriptsParams(
        transcribe_audio_chunks_params=transcribe_audio_chunks_params,
        llm_model_id=llm_model_id,
        llm_system_and_user_prompt_to_response=models[llm_model_id][use_llm_model_key],
        length_factor=3,
        summary_format="""
        Key Insights:
        
        * <sentence 1>
        * <sentence 2>
        ...
        * <sentence {{num_sentences}}>
        """,
    )
    for llm_model_id in models
]

faithful_and_required_format_summaries: List[SummarizeTranscriptsParams] = []
for summarize_transcripts_params in summarize_transcripts_params_list:
    print(f"handle {summarize_transcripts_params.llm_model_id}...")

    summarize_transcripts.summarize(
        summarize_transcripts_params=summarize_transcripts_params,
    )

    evaluate_summary_params = EvaluateSummaryParams(
        summarize_transcripts_params=summarize_transcripts_params,
        llm_evaluator_system_and_user_prompt_to_response=models[gpt_5_mini_model_key][
            use_llm_model_key
        ],
    )

    evaluate_summary_format.evaluate(evaluate_summary_params=evaluate_summary_params)

    evaluate_summary_faithfulness.evaluate(
        evaluate_summary_params=evaluate_summary_params
    )

    is_faithful = evaluate_summary_utils.is_faithful(evaluate_summary_params)
    is_in_required_format = evaluate_summary_utils.is_in_required_format(
        evaluate_summary_params
    )

    if is_faithful and is_in_required_format:
        faithful_and_required_format_summaries.append(summarize_transcripts_params)

print("models for competition:")
[print(summary.llm_model_id) for summary in faithful_and_required_format_summaries]

# compare summaries

compare_summaries_params = CompareSummariesParams(
    comparison_path="./",
    transcribe_audio_chunks_params=transcribe_audio_chunks_params,
    summaries=faithful_and_required_format_summaries,
    llm_evaluator_system_and_user_prompt_to_response=models[gpt_5_mini_model_key][
        use_llm_model_key
    ],
)

compare_summaries.compare(compare_summaries_params=compare_summaries_params)

compare_summaries.calculate_points(compare_summaries_params=compare_summaries_params)
