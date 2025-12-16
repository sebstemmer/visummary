from download_video.download_video_params import DownloadVideoParams


class ChunkAudioParams:
    def __init__(
        self,
        download_video_params: DownloadVideoParams,
        chunk_size_in_min: int,
        overlap_in_percent: int,
    ):
        self.base_path = download_video_params.base_path
        self.chunk_size_in_min = chunk_size_in_min
        self.overlap_in_percent = overlap_in_percent
