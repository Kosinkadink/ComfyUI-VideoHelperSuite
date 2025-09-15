import os
import itertools
import numpy as np
import torch
import cv2
import psutil
import subprocess
import re
import time

from comfy.utils import common_upscale, ProgressBar
from comfy.k_diffusion.utils import FolderOfImages

import folder_paths
import nodes

from .logger import logger
from .utils import (
    BIGMAX, DIMMAX, calculate_file_hash, get_sorted_dir_files_from_directory,
    lazy_get_audio, hash_path, validate_path, strip_path, try_download_video,
    is_url, imageOrLatent, ffmpeg_path, ENCODE_ARGS, floatOrInt
)

# ------------------------------
# Constants / Formats
# ------------------------------

video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']

VHSLoadFormats = {
    'None': {},
    'AnimateDiff': {'target_rate': 8, 'dim': (8,0,512,512)},
    'Mochi': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(6,1)},
    'LTXV': {'target_rate': 24, 'dim': (32,0,768,512), 'frames':(8,1)},
    'Hunyuan': {'target_rate': 24, 'dim': (16,0,848,480), 'frames':(4,1)},
    'Cosmos': {'target_rate': 24, 'dim': (16,0,1280,704), 'frames':(8,1)},
    'Wan': {'target_rate': 16, 'dim': (8,0,832,480), 'frames':(4,1)},
}
"""
External plugins may add additional formats to nodes.VHSLoadFormats
In addition to shorthand options, direct widget names will map a given dict to options.
Adding a third arguement to a frames tuple can enable strict checks on number
of loaded frames, i.e (8,1,True)
"""
if not hasattr(nodes, 'VHSLoadFormats'):
    nodes.VHSLoadFormats = {}

def get_load_formats():
    formats = {}
    formats.update(nodes.VHSLoadFormats)
    formats.update(VHSLoadFormats)
    return (list(formats.keys()),
            {'default': 'AnimateDiff', 'formats': formats})

def get_format(format):
    if format in VHSLoadFormats:
        return VHSLoadFormats[format]
    return nodes.VHSLoadFormats.get(format, {})

def is_gif(filename) -> bool:
    file_parts = filename.split('.')
    return len(file_parts) > 1 and file_parts[-1].lower() == "gif"

# ------------------------------
# Helpers
# ------------------------------

def target_size(width, height, custom_width, custom_height, downscale_ratio=8) -> tuple[int, int]:
    if downscale_ratio is None:
        downscale_ratio = 8
    w, h = float(width), float(height)
    if custom_width == 0 and custom_height == 0:
        pass
    elif custom_height == 0:
        h = h * (custom_width / w)
        w = custom_width
    elif custom_width == 0:
        w = w * (custom_height / h)
        h = custom_height
    else:
        w = custom_width
        h = custom_height
    w = int(w / downscale_ratio + 0.5) * downscale_ratio
    h = int(h / downscale_ratio + 0.5) * downscale_ratio
    return (int(w), int(h))

#Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
def batched(it, n):
    while batch := tuple(itertools.islice(it, n)):
        yield batch

def batched_vae_encode(images, vae, frames_per_batch):
    for batch in batched(images, frames_per_batch):
        image_batch = torch.from_numpy(np.ascontiguousarray(batch))  # [B,H,W,C], float32 [0,1]
        out = vae.encode(image_batch)
        if isinstance(out, torch.Tensor):
            yield from out.detach().cpu().numpy()
        elif isinstance(out, dict) and "samples" in out and isinstance(out["samples"], torch.Tensor):
            yield from out["samples"].detach().cpu().numpy()
        else:
            yield from np.asarray(out)

# ------------------------------
# CPU-side pre-tensor resize helpers (OpenCV path)
# ------------------------------

def _resize_frame_np(frame: np.ndarray, dst_w: int, dst_h: int, mode: str) -> np.ndarray:
    """
    frame: [H,W,C] float32 [0,1], C=3 or 4
    mode: 'fill' (cover crop center), 'fit' (letterbox), 'stretch'
    """
    if dst_w <= 0 or dst_h <= 0:
        return frame
    h, w = frame.shape[:2]
    if mode == "stretch":
        return cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_LANCZOS4)

    src_ar = w / h
    dst_ar = dst_w / dst_h

    if mode in ("fill", "crop"):  # cover: scale up, then center-crop
        scale = max(dst_w / w, dst_h / h)
        new_w, new_h = int(np.round(w * scale)), int(np.round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        x0 = (new_w - dst_w) // 2
        y0 = (new_h - dst_h) // 2
        return resized[y0:y0 + dst_h, x0:x0 + dst_w]

    # fit / contain: scale down, then pad
    scale = min(dst_w / w, dst_h / h)
    new_w, new_h = int(np.round(w * scale)), int(np.round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    pad = np.zeros((dst_h, dst_w, frame.shape[2]), dtype=frame.dtype)
    # pad uses zeros (black; alpha=0 if present)
    x0 = (dst_w - new_w) // 2
    y0 = (dst_h - new_h) // 2
    pad[y0:y0 + new_h, x0:x0 + new_w] = resized
    return pad

# ------------------------------
# OpenCV-based frame generator
# ------------------------------

def cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                       select_every_nth, meta_batch=None, unique_id=None):
    video_cap = cv2.VideoCapture(video)
    if not video_cap.isOpened() or not video_cap.grab():
        raise ValueError(f"{video} could not be loaded with cv.")

    fps = video_cap.get(cv2.CAP_PROP_FPS) or 1.0
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    if width <= 0 or height <= 0:
        _, frame = video_cap.retrieve()
        if frame is None:
            raise ValueError(f"{video} no readable frames.")
        height, width, _ = frame.shape

    base_frame_time = 1.0 / fps if fps > 0 else 0.0
    target_frame_time = base_frame_time if (force_rate == 0) else (1.0 / float(force_rate))

    if total_frames > 0:
        yieldable_frames = int(duration * (force_rate or fps))
        if select_every_nth:
            yieldable_frames //= select_every_nth
        if frame_load_cap != 0:
            yieldable_frames = min(frame_load_cap, yieldable_frames)
    else:
        yieldable_frames = 0

    yield (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames)

    pbar = ProgressBar(yieldable_frames)
    total_frame_count = 0
    total_frames_evaluated = -1
    frames_added = 0
    prev_frame = None
    time_offset = target_frame_time

    while video_cap.isOpened():
        if time_offset < target_frame_time:
            is_returned = video_cap.grab()
            if not is_returned:
                break
            time_offset += base_frame_time
        if time_offset < target_frame_time:
            continue
        time_offset -= target_frame_time

        total_frame_count += 1
        if total_frame_count <= max(0, int(skip_first_frames)):
            continue
        else:
            total_frames_evaluated += 1

        if select_every_nth and (total_frames_evaluated % select_every_nth != 0):
            continue

        ok, frame = video_cap.retrieve()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0

        if prev_frame is not None:
            inp = yield prev_frame
            if inp is not None:
                return
        prev_frame = frame
        frames_added += 1
        if pbar is not None:
            pbar.update_absolute(frames_added, yieldable_frames)

        if frame_load_cap > 0 and frames_added >= frame_load_cap:
            break

    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id, None)
        meta_batch.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

# ------------------------------
# FFMPEG-based frame generator (with resize modes)
# ------------------------------

def ffmpeg_frame_generator(video, force_rate, frame_load_cap, start_time,
                           custom_width, custom_height, downscale_ratio=8,
                           resize_mode="auto",
                           meta_batch=None, unique_id=None):
    args_input = ["-i", video]
    args_dummy = [ffmpeg_path] + args_input + ['-c', 'copy', '-frames:v', '1', "-f", "null", "-"]
    try:
        dummy_res = subprocess.run(args_dummy, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the ffmpeg subprocess:\n" + e.stderr.decode(*ENCODE_ARGS))
    lines = dummy_res.stderr.decode(*ENCODE_ARGS)

    if "Video: vp9 " in lines:
        args_input = ["-c:v", "libvpx-vp9"] + args_input
        args_dummy = [ffmpeg_path] + args_input + ['-c', 'copy', '-frames:v', '1', "-f", "null", "-"]
        try:
            dummy_res = subprocess.run(args_dummy, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            raise Exception("An error occurred in the ffmpeg subprocess:\n" + e.stderr.decode(*ENCODE_ARGS))
        lines = dummy_res.stderr.decode(*ENCODE_ARGS)

    size_base = None
    fps_base = None
    alpha = False
    for line in lines.split('\n'):
        match = re.search(r"^ *Stream .* Video.*, ([1-9]|\d{2,})x(\d+)", line)
        if match is not None:
            size_base = [int(match.group(1)), int(match.group(2))]
            fps_match = re.search(r", ([\d\.]+) fps", line)
            fps_base = float(fps_match.group(1)) if fps_match else 1.0
            alpha = re.search(r"(yuva|rgba|bgra)", line) is not None
            break
    if size_base is None:
        raise Exception("Failed to parse video/image information. FFMPEG output:\n" + lines)

    durs_match = re.search(r"Duration: (\d+:\d+:\d+\.\d+),", lines)
    if durs_match:
        durs = durs_match.group(1).split(':')
        duration = int(durs[0]) * 360 + int(durs[1]) * 60 + float(durs[2])
    else:
        duration = 0.0

    if start_time > 0:
        if start_time > 4:
            post_seek = ['-ss', '4']
            args_input = ['-ss', str(start_time - 4)] + args_input
        else:
            post_seek = ['-ss', str(start_time)]
    else:
        post_seek = []
    args_all_frames = [ffmpeg_path, "-v", "error", "-an"] + args_input + ["-pix_fmt", "rgba64le"] + post_seek

    vfilters = []
    if force_rate != 0:
        vfilters.append(f"fps=fps={force_rate}")

    # Resizing path
    if custom_width != 0 or custom_height != 0:
        size = target_size(size_base[0], size_base[1], custom_width, custom_height, downscale_ratio=downscale_ratio)
        dst_w, dst_h = size
        if resize_mode == "auto":
            resize_mode_eff = "fill" if (custom_width and custom_height) else "stretch"
        else:
            resize_mode_eff = resize_mode

        if resize_mode_eff in ("fill", "crop"):
            # cover: crop to AR, then scale
            ar = float(dst_w) / float(dst_h)
            vfilters.append(f"crop=if(gt({ar}\\,a)\\,iw\\,ih*{ar}):if(gt({ar}\\,a)\\,iw/{ar}\\,ih)")
            vfilters.append(f"scale={dst_w}:{dst_h}")
        elif resize_mode_eff == "fit":
            # contain: scale with aspect, then pad to exact
            # force_original_aspect_ratio=decrease keeps AR; then pad to WxH centered
            vfilters.append(f"scale={dst_w}:{dst_h}:force_original_aspect_ratio=decrease")
            # pad color: black (works for rgba64le too; alpha=1)
            vfilters.append(f"pad={dst_w}:{dst_h}:(ow-iw)/2:(oh-ih)/2")
        else:  # stretch
            vfilters.append(f"scale={dst_w}:{dst_h}")
    else:
        size = tuple(size_base)

    if len(vfilters) > 0:
        args_all_frames += ["-vf", ",".join(vfilters)]

    yieldable_frames = (force_rate or fps_base) * duration
    if frame_load_cap > 0:
        args_all_frames += ["-frames:v", str(frame_load_cap)]
        yieldable_frames = min(yieldable_frames, frame_load_cap)
    yield (size_base[0], size_base[1], fps_base, duration, fps_base * duration,
           1 / (force_rate or fps_base), yieldable_frames, size[0], size[1], alpha)

    args_all_frames += ["-f", "rawvideo", "-"]
    pbar = ProgressBar(yieldable_frames)
    try:
        with subprocess.Popen(args_all_frames, stdout=subprocess.PIPE) as proc:
            bpi = size[0] * size[1] * 8
            current_bytes = bytearray(bpi)
            current_offset = 0
            prev_frame = None
            while True:
                bytes_read = proc.stdout.read(bpi - current_offset)
                if bytes_read is None:
                    time.sleep(.1)
                    continue
                if len(bytes_read) == 0:
                    break
                current_bytes[current_offset:current_offset+len(bytes_read)] = bytes_read
                current_offset += len(bytes_read)
                if current_offset == bpi:
                    if prev_frame is not None:
                        yield prev_frame
                        pbar.update(1)
                    frame16 = np.frombuffer(current_bytes, dtype=np.dtype(np.uint16).newbyteorder("<"))
                    frame16 = frame16.reshape(size[1], size[0], 4)
                    frame = (frame16.astype(np.float32) / (2**16 - 1))
                    if not alpha:
                        frame = frame[:, :, :3]
                    prev_frame = frame
                    current_offset = 0
    except BrokenPipeError:
        raise Exception("An error occured in the ffmpeg subprocess.")
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id, None)
        meta_batch.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

# ------------------------------
# Resize-on-the-fly CV generator (with resize modes)
# ------------------------------

def resized_cv_frame_gen(custom_width, custom_height, downscale_ratio, resize_mode="auto", **kwargs):
    gen = cv_frame_generator(**kwargs)
    info = next(gen)
    width, height = info[0], info[1]
    frames_per_batch = (1920 * 1080 * 16) // max(1, (width * height)) or 1
    if kwargs.get('meta_batch', None) is not None:
        frames_per_batch = min(frames_per_batch, kwargs['meta_batch'].frames_per_batch)

    need_resize = (custom_width != 0 or custom_height != 0 or downscale_ratio is not None)
    new_size = target_size(width, height, custom_width, custom_height, downscale_ratio) if need_resize else (width, height)
    dst_w, dst_h = new_size

    # Determine effective mode
    if resize_mode == "auto":
        resize_mode_eff = "fill" if (custom_width and custom_height) else "stretch"
    else:
        resize_mode_eff = resize_mode

    yield (*info, new_size[0], new_size[1], False)

    if need_resize and (new_size[0] != width or new_size[1] != height or resize_mode_eff != "stretch"):
        # Do pre-tensor resizing in NumPy with OpenCV
        def rescale_batch(batch):
            out = []
            for f in batch:
                out.append(_resize_frame_np(f, dst_w, dst_h, mode=resize_mode_eff))
            return np.asarray(out, dtype=np.float32)

        for small_batch in batched(gen, frames_per_batch):
            resized = rescale_batch(small_batch)
            for f in resized:
                yield f
        return

    # Stretch with cv path happens implicitly below if only one side forced
    yield from gen

# ------------------------------
# Main loader
# ------------------------------

def load_video(meta_batch=None, unique_id=None, memory_limit_mb=None, vae=None,
               generator=resized_cv_frame_gen, format='None',  **kwargs):
    if 'force_size' in kwargs:
        kwargs.pop('force_size')
        logger.warn("force_size has been removed. Did you reload the webpage after updating?")
    fmt = get_format(format)
    kwargs['video'] = strip_path(kwargs['video'])

    if vae is not None:
        downscale_ratio = getattr(vae, "downscale_ratio", 8)
    else:
        downscale_ratio = fmt.get('dim', (1,))[0]

    # Prime generator / reuse
    if meta_batch is None or unique_id not in (meta_batch.inputs if meta_batch else {}):
        gen = generator(meta_batch=meta_batch, unique_id=unique_id, downscale_ratio=downscale_ratio, **kwargs)
        (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames,
         new_width, new_height, alpha) = next(gen)

        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, width, height, fps, duration, total_frames,
                                            target_frame_time, yieldable_frames, new_width, new_height, alpha)
            if yieldable_frames:
                meta_batch.total_frames = min(meta_batch.total_frames, yieldable_frames)
    else:
        (gen, width, height, fps, duration, total_frames, target_frame_time, yieldable_frames,
         new_width, new_height, alpha) = meta_batch.inputs[unique_id]

    if memory_limit_mb is not None:
        memory_limit = int(memory_limit_mb) * (2 ** 20)
    else:
        try:
            memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - (2 ** 27)
        except Exception:
            logger.warn("Failed to calculate available memory. Memory load limit has been disabled")
            memory_limit = BIGMAX

    if vae is not None:
        max_loadable_frames = int(memory_limit // (width * height * 3 * (4 + 4 + 0.1)))
    else:
        max_loadable_frames = int(memory_limit // max(1, (width * height * 3 * 0.1)))

    if meta_batch is not None:
        if 'frames' in fmt:
            mod_rule = fmt['frames']
            if meta_batch.frames_per_batch % mod_rule[0] != mod_rule[1]:
                error = (meta_batch.frames_per_batch - mod_rule[1]) % mod_rule[0]
                suggested = meta_batch.frames_per_batch - error
                if error > mod_rule[0] / 2:
                    suggested += mod_rule[0]
                raise RuntimeError(f"The chosen frames per batch is incompatible with the selected format. Try {suggested}")
        if meta_batch.frames_per_batch > max_loadable_frames:
            raise RuntimeError(f"Meta Batch set to {meta_batch.frames_per_batch} frames but only {max_loadable_frames} can fit in memory")
        gen = itertools.islice(gen, meta_batch.frames_per_batch)
    else:
        original_gen = gen
        gen = itertools.islice(gen, max_loadable_frames)

    if vae is not None:
        frames = list(batched_vae_encode(gen, vae, frames_per_batch=(1920 * 1080 * 16) // max(1, (width * height)) or 1))
        if len(frames) == 0:
            raise RuntimeError("No frames generated")
        arr = np.ascontiguousarray(frames, dtype=np.float32)
        images = torch.from_numpy(arr)
        result_images = {"samples": images}
    else:
        frames = list(gen)
        if len(frames) == 0:
            raise RuntimeError("No frames generated")
        arr = np.ascontiguousarray(np.stack(frames, axis=0))
        images = torch.from_numpy(arr)
        result_images = images

    if meta_batch is None and memory_limit is not None:
        try:
            next(original_gen)
            raise RuntimeError(f"Memory limit hit after loading {len(images) if isinstance(images, torch.Tensor) else len(images['samples'])} frames. Stopping execution.")
        except StopIteration:
            pass

    if 'frames' in fmt:
        div, mod = fmt['frames'][:2]
        strict = (len(fmt['frames']) > 2 and fmt['frames'][2])
        total_loaded = (len(images) if isinstance(images, torch.Tensor) else len(images["samples"]))
        if total_loaded % div != mod:
            err_msg = f"The number of frames loaded {total_loaded}, does not match the requirements of the currently selected format."
            if strict:
                raise RuntimeError(err_msg)
            frames_ok = (total_loaded - mod) // div * div + mod
            if isinstance(images, torch.Tensor):
                images = images[:frames_ok]
                result_images = images
            else:
                images["samples"] = images["samples"][:frames_ok]
                result_images = images

    if 'start_time' in kwargs:
        start_time = kwargs['start_time']
    else:
        start_time = kwargs['skip_first_frames'] * target_frame_time
    target_frame_time *= kwargs.get('select_every_nth', 1)

    audio = lazy_get_audio(kwargs['video'], start_time, kwargs['frame_load_cap'] * target_frame_time)

    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1 / target_frame_time if target_frame_time > 0 else 0,
        "loaded_frame_count": (len(images) if isinstance(images, torch.Tensor) else len(images["samples"])),
        "loaded_duration": (len(images) if isinstance(images, torch.Tensor) else len(images["samples"])) * target_frame_time,
        "loaded_width": new_width,
        "loaded_height": new_height,
    }

    if vae is None:
        return (result_images, len(images), audio, video_info)
    else:
        return (result_images, len(result_images["samples"]), audio, video_info)

# ------------------------------
# Nodes (added resize_mode to INPUT_TYPES)
# ------------------------------

_RESIZE_CHOICES = ["auto", "fill", "fit", "stretch", "crop"]

class LoadVideoUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1].lower() in video_extensions):
                    files.append(f)
        return {
            "required": {
                "video": (sorted(files),),
                "force_rate": (floatOrInt, {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "resize_mode": (_RESIZE_CHOICES,),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": get_load_formats(),
            },
            "hidden": {
                "force_size": "STRING",
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = (imageOrLatent, "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")

    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(strip_path(kwargs['video']))
        return load_video(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return calculate_file_hash(image_path)

    @classmethod
    def VALIDATE_INPUTS(s, video):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True


class LoadVideoPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"placeholder": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "force_rate": (floatOrInt, {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "resize_mode": ("STRING", {"default": "auto", "choices": _RESIZE_CHOICES}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": get_load_formats(),
            },
            "hidden": {
                "force_size": "STRING",
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = (imageOrLatent, "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")

    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        if is_url(kwargs['video']):
            kwargs['video'] = try_download_video(kwargs['video']) or kwargs['video']
        return load_video(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video):
        return validate_path(video, allow_none=True)


class LoadVideoFFmpegUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1].lower() in video_extensions):
                    files.append(f)
        return {
            "required": {
                "video": (sorted(files),),
                "force_rate": (floatOrInt, {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "resize_mode": ("STRING", {"default": "auto", "choices": _RESIZE_CHOICES}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0}),
                "start_time": ("FLOAT", {"default": 0, "min": 0, "max": BIGMAX, "step": .001, "widgetType": "VHSTIMESTAMP"}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": get_load_formats(),
            },
            "hidden": {
                "force_size": "STRING",
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = (imageOrLatent, "MASK", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "mask", "audio", "video_info")

    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(strip_path(kwargs['video']))
        image, _, audio, video_info = load_video(**kwargs, generator=ffmpeg_frame_generator)
        if isinstance(image, dict):
            return (image, None, audio, video_info)
        if image.size(3) == 4:
            return (image[:, :, :, :3], 1 - image[:, :, :, 3], audio, video_info)
        return (image, torch.zeros(image.size(0), 64, 64, device="cpu"), audio, video_info)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return calculate_file_hash(image_path)

    @classmethod
    def VALIDATE_INPUTS(s, video):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True


class LoadVideoFFmpegPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"placeholder": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "force_rate": (floatOrInt, {"default": 0, "min": 0, "max": 60, "step": 1, "disable": 0}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, 'disable': 0}),
                "resize_mode": ("STRING", {"default": "auto", "choices": _RESIZE_CHOICES}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1, "disable": 0}),
                "start_time": ("FLOAT", {"default": 0, "min": 0, "max": BIGMAX, "step": .001, "widgetType": "VHSTIMESTAMP"}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": get_load_formats(),
            },
            "hidden": {
                "force_size": "STRING",
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = (imageOrLatent, "MASK", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "mask", "audio", "video_info")

    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        if is_url(kwargs['video']):
            kwargs['video'] = try_download_video(kwargs['video']) or kwargs['video']
        image, _, audio, video_info = load_video(**kwargs, generator=ffmpeg_frame_generator)
        if isinstance(image, dict):
            return (image, None, audio, video_info)
        if image.size(3) == 4:
            return (image[:, :, :, :3], 1 - image[:, :, :, 3], audio, video_info)
        return (image, torch.zeros(image.size(0), 64, 64, device="cpu"), audio, video_info)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video):
        return validate_path(video, allow_none=True)


class LoadImagePath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("STRING", {"placeholder": "X://insert/path/here.png", "vhs_path_extensions": list(FolderOfImages.IMG_EXTENSIONS)}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": DIMMAX, "step": 8, 'disable': 0}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": DIMMAX, "step": 8, 'disable': 0}),
                "resize_mode": ("STRING", {"default": "auto", "choices": _RESIZE_CHOICES}),
            },
            "optional": {
                "vae": ("VAE",),
            },
            "hidden": {
                "force_size": "STRING",
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = (imageOrLatent, "MASK")
    RETURN_NAMES = ("IMAGE", "mask")

    FUNCTION = "load_image"

    def load_image(self, **kwargs):
        if kwargs['image'] is None or validate_path(kwargs['image']) != True:
            raise Exception("image is not a valid path: " + kwargs['image'])
        kwargs.update({'video': kwargs['image'], 'force_rate': 0, 'frame_load_cap': 0, 'start_time': 0})
        kwargs.pop('image')
        image, _, _, _ = load_video(**kwargs, generator=ffmpeg_frame_generator)
        if isinstance(image, dict):
            return (image, None)
        if image.size(3) == 4:
            return (image[:, :, :, :3], 1 - image[:, :, :, 3])
        return (image, torch.zeros(image.size(0), 64, 64, device="cpu"))
