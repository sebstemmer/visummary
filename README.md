# Summarizing YouTube Videos Locally Using Whisper and LLMs

This repository contains the end-to-end pipeline to summarize YouTube videos locally on an **Apple M2 Max with 96 GB of
RAM**. The pipeline downloads YouTube videos with [yt-dlp](https://github.com/yt-dlp/yt-dlp), chunks the audio
with [pydub](https://github.com/jiaaro/pydub), transcribes the audio chunks
with [OpenAI’s Whisper](https://en.wikipedia.org/wiki/Whisper_(speech_recognition_system))
via [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
and summarizes the transcript with LLMs via [Ollama](https://ollama.com/). It includes the evaluation of different
Whisper and LLM models, such
as [Qwen 3](https://ollama.com/library/qwen3), [Qwen 2.5](https://ollama.com/library/qwen2.5), [Llama 3](https://ollama.com/library/llama3), [DeepSeek R1](https://ollama.com/library/deepseek-r1), [Gemma 3](https://ollama.com/library/gemma3),
and [gpt-oss](https://ollama.com/library/gpt-oss), from an AI engineering perspective. A detailed explanation and
additional context can be found
in [a post on my personal blog](https://sebstemmer.com/ai/engineering/2026/01/17/summarizing-youtube-videos-locally-using-whisper-and-llms.html).

**Note:** All scripts should be executed using

```
python -m ...
```

from the **root** of the repository.

All scripts and functions log relevant information to the console.

## Structure

```
├── data/
├── analysis/
├── common/
├── pipeline/
```

* `data/` contains all data produced by the pipeline and the evaluation, including the audio files, audio chunks,
  transcripts, and summaries.
* `analysis/` contains everything required for the evaluation of the Whisper models and LLMs.
* `pipeline/` contains the end-to-end pipeline for summarizing videos locally.
* `common/` contains functionality such as chunking audio files or downloading videos that are needed by the pipeline
  and its analysis.

## Installation

After cloning the repository, create a virtual environment:

```
python -m venv venv
```

Activate the environment:

```
source ./venv/bin/activate
```

Install the required dependencies:

```
pip install -r requirements.txt
```

The library `pydub` requires [FFmpeg](https://www.ffmpeg.org/). On macOS, you can install it with Homebrew:

```
brew install ffmpeg
```

The end-to-end pipeline also requires [Ollama](https://ollama.com/). Make sure that your Ollama installation uses the
full capacity of your machine. For example, if your Mac
supports [MPS](https://developer.apple.com/documentation/metalperformanceshaders), install Ollama directly on your
machine.

## End-to-End Pipeline

The pipeline can be executed with:

```
python -m pipeline.main
```

It will print out the summary in the format

```
Key Insights:

* This is the first sentence of the summary.
* ...
* This is the nth sentence of the summary.
```

### Download a YouTube Video

You can download a YouTube video with:

```
common/download_video/download_video.py - download
```

The video is downloaded in YouTube's native `.webm` format.

### Chunk the Audio

The audio is chunked using:

```
common/chunk_audio/chunk_audio.py - chunk
```

The chunks have a size of `chunk_size_in_min` and overlap between consecutive chunks by `overlap_in_percent`. They are
persisted in `.mp3` format.

### Transcribe the Audio Chunks

The audio chunks are transcribed using:

```
common/transcribe_audio_chunks/transcribe_audio_chunks.py - transcribe
```

The resulting transcript snippets are persisted in `.txt` format.

### Summarize the Transcript Snippets

The transcript snippets are summarized using:

```
common/summarize_transcript/summarize_transcript.py - summarize
```

The `length_factor` determines the length of the summaries. The summaries are persisted in JSON format.

## Evaluation

The evaluation of the Whisper models is started with the script:

```
python -m analyze.analyze_different_stts.main
```

The script can be restarted because it persists data under the `data` folder and therefore knows what has already been
processed.

It computes the quality score and speed factor and outputs the models sorted by these metrics.

The evaluation of the LLMs is started with the script:

```
python -m analyze.analyze_different_llms.main
```

The script persists intermediate results under the `data` folder, allowing it to be restarted.

The script reports all models that adhere to the required output format and computes the coverage score, faithfulness
score, combined score, and LLM speed factor, sorting the models accordingly.

