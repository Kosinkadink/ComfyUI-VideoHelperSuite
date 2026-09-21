import math
from pathlib import Path

MAX_AUDIO_SECONDS = 60
MAX_VIDEO_FRAMES = 512
MAX_VIDEO_PIXELS = 3_000_000


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def available_memory_bytes(
    host_available: int,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    reserve_bytes: int = 2**27,
) -> int:
    available = host_available
    for limit_name, usage_name in (
        ("memory.max", "memory.current"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes"),
    ):
        limit = _read_int(cgroup_root / limit_name)
        usage = _read_int(cgroup_root / usage_name)
        if limit is not None and usage is not None:
            available = min(available, max(0, limit - usage))
            break
    return max(0, available - reserve_bytes)


def max_loadable_frames(
    memory_bytes: int,
    width: int,
    height: int,
    channels: int,
    bytes_per_channel: float = 4,
) -> int:
    return int(memory_bytes // (width * height * channels * bytes_per_channel))


def effective_frame_cap(requested: int, maximum: int) -> int:
    return min(requested or maximum, maximum) if maximum > 0 else requested


def bounded_audio_duration(frame_count: int, frame_time: float, maximum: float) -> float:
    if not math.isfinite(frame_time) or frame_time <= 0:
        raise ValueError(f"Invalid frame time for audio extraction: {frame_time}")
    return min(frame_count * frame_time, maximum)


def validate_video_pixels(
    source_width: int,
    source_height: int,
    loaded_width: int,
    loaded_height: int,
    maximum: int,
) -> None:
    if maximum > 0 and (source_width * source_height > maximum or loaded_width * loaded_height > maximum):
        raise ValueError(
            f"Video resolution exceeds the {maximum:,}-pixel limit: "
            f"source={source_width}x{source_height}, loaded={loaded_width}x{loaded_height}"
        )
