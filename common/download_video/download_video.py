import os

import yt_dlp


def download(audio_path: str, video_url: str) -> None:
    """
    Downloads the video at the given URL as a .webm file.

    Args:
        audio_path (str):
            Path where the downloaded video will be saved.
        video_url (str):
            URL of the video.

    Returns:
        None
    """
    print(f"downloading video...")

    # check if already done

    if os.path.isfile(audio_path):
        print(f"...video already downloaded")
        return

    # create folders if they do not exist yet

    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    # download video as webm

    opts = {
        "format": "bestaudio[ext=webm]",
        f"outtmpl": audio_path,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([video_url])

    print(f"...downloaded video")
