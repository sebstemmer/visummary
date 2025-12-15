from typing import NamedTuple

class ChunkAudioParams(NamedTuple):
    chunk_size_in_min: int
    overlap_in_percent: int