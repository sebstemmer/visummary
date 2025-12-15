import os
import yt_dlp
from download_video import download_video_utils


def download_video(video_url: str, base_path: str):
    print(f"downloading video {video_url}...")

    audio_path = download_video_utils.get_audio_path(base_path)

    if os.path.isfile(audio_path):
        print(f"...video {video_url} already downloaded")
        return

    opts = {
        "format": "bestaudio[ext=webm]",
        f"outtmpl": audio_path,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([video_url])

    print(f"...{video_url} video downloaded")
