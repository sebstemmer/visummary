import os
import yt_dlp
from download_video import download_video_utils
from download_video.download_video_params import DownloadVideoParams


def download(download_video_params: DownloadVideoParams) -> None:
    print(f"downloading video...")

    audio_path = download_video_utils.get_audio_path(download_video_params.base_path)

    if os.path.isfile(audio_path):
        print(f"...video already downloaded")
        return

    opts = {
        "format": "bestaudio[ext=webm]",
        f"outtmpl": audio_path,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([download_video_params.video_url])

    print(f"...downloaded video")
