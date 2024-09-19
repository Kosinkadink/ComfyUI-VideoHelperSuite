import os
import itertools
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import psutil

import folder_paths
from comfy.utils import common_upscale, ProgressBar
from .logger import logger
from .utils import BIGMAX, DIMMAX, calculate_file_hash, get_sorted_dir_files_from_directory,\
        lazy_get_audio, hash_path, validate_path, strip_path, try_download_video, is_url, imageOrLatent


video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov']


def is_gif(filename) -> bool:
    file_parts = filename.split('.')
    return len(file_parts) > 1 and file_parts[-1] == "gif"


def target_size(width, height, force_size, custom_width, custom_height, downscale_ratio=8) -> tuple[int, int]:
    if force_size == "Disabled":
        pass
    elif force_size == "Custom Width" or force_size.endswith('x?'):
        height *= custom_width/width
        width = custom_width
    elif force_size == "Custom Height" or force_size.startswith('?x'):
        width *= custom_height/height
        height = custom_height
    else:
        width = custom_width
        height = custom_height
    width = int(width/downscale_ratio + 0.5) * downscale_ratio
    height = int(height/downscale_ratio + 0.5) * downscale_ratio
    return (width, height)

def cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                       select_every_nth, meta_batch=None, unique_id=None):
    video_cap = cv2.VideoCapture(strip_path(video))
    if not video_cap.isOpened():
        raise ValueError(f"{video} could not be loaded with cv.")

    # extract video metadata
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # set video_cap to look at start_index frame
    total_frame_count = 0
    total_frames_evaluated = -1
    frames_added = 0
    base_frame_time = 1 / fps
    prev_frame = None

    if force_rate == 0:
        target_frame_time = base_frame_time
    else:
        target_frame_time = 1/force_rate

    yield (width, height, fps, duration, total_frames, target_frame_time)
    if total_frames > 0:
        if force_rate != 0:
            yieldable_frames = int(total_frames / fps * force_rate)
        else:
            yieldable_frames = total_frames
        if frame_load_cap != 0:
            yieldable_frames =  min(frame_load_cap, yieldable_frames)
    else:
        yieldable_frames = 0
    pbar = ProgressBar(yieldable_frames)
    if meta_batch is not None:
        yield yieldable_frames
    time_offset=target_frame_time - base_frame_time
    while video_cap.isOpened():
        if time_offset < target_frame_time:
            is_returned = video_cap.grab()
            # if didn't return frame, video has ended
            if not is_returned:
                break
            time_offset += base_frame_time
        if time_offset < target_frame_time:
            continue
        time_offset -= target_frame_time
        # if not at start_index, skip doing anything with frame
        total_frame_count += 1
        if total_frame_count <= skip_first_frames:
            continue
        else:
            total_frames_evaluated += 1

        # if should not be selected, skip doing anything with frame
        if total_frames_evaluated%select_every_nth != 0:
            continue

        # opencv loads images in BGR format (yuck), so need to convert to RGB for ComfyUI use
        # follow up: can videos ever have an alpha channel?
        # To my testing: No. opencv has no support for alpha
        unused, frame = video_cap.retrieve()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert frame to comfyui's expected format
        # TODO: frame contains no exif information. Check if opencv2 has already applied
        frame = np.array(frame, dtype=np.float32)
        torch.from_numpy(frame).div_(255)
        if prev_frame is not None:
            inp  = yield prev_frame
            if inp is not None:
                #ensure the finally block is called
                return
        prev_frame = frame
        frames_added += 1
        if pbar is not None:
            pbar.update_absolute(frames_added, yieldable_frames)
        # if cap exists and we've reached it, stop processing frames
        if frame_load_cap > 0 and frames_added >= frame_load_cap:
            break
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

#Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
def batched(it, n):
    while batch := tuple(itertools.islice(it, n)):
        yield batch
def batched_vae_encode(images, vae, frames_per_batch):
    for batch in batched(images, frames_per_batch):
        image_batch = torch.from_numpy(np.array(batch))
        yield from vae.encode(image_batch).numpy()

def load_video_cv(video: str, force_rate: int, force_size: str,
                  custom_width: int,custom_height: int, frame_load_cap: int,
                  skip_first_frames: int, select_every_nth: int,
                  meta_batch=None, unique_id=None, memory_limit_mb=None, vae=None):
    if meta_batch is None or unique_id not in meta_batch.inputs:
        gen = cv_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                                 select_every_nth, meta_batch, unique_id)
        (width, height, fps, duration, total_frames, target_frame_time) = next(gen)

        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, width, height, fps, duration, total_frames, target_frame_time)
            yieldable_frames = next(gen)
            if yieldable_frames:
                meta_batch.total_frames = min(meta_batch.total_frames, yieldable_frames)

    else:
        (gen, width, height, fps, duration, total_frames, target_frame_time) = meta_batch.inputs[unique_id]

    memory_limit = None
    if memory_limit_mb is not None:
        memory_limit *= 2 ** 20
    else:
        #TODO: verify if garbage collection should be performed here.
        #leaves ~128 MB unreserved for safety
        try:
            memory_limit = (psutil.virtual_memory().available + psutil.swap_memory().free) - 2 ** 27
        except:
            logger.warn("Failed to calculate available memory. Memory load limit has been disabled")
    if memory_limit is not None:
        if vae is not None:
            #space required to load as f32, exist as latent with wiggle room, decode to f32
            max_loadable_frames = int(memory_limit//(width*height*3*(4+4+1/10)))
        else:
            #TODO: use better estimate for when vae is not None
            #Consider completely ignoring for load_latent case?
            max_loadable_frames = int(memory_limit//(width*height*3*(.1)))
        if meta_batch is not None:
            if meta_batch.frames_per_batch > max_loadable_frames:
                raise RuntimeError(f"Meta Batch set to {meta_batch.frames_per_batch} frames but only {max_loadable_frames} can fit in memory")
            gen = itertools.islice(gen, meta_batch.frames_per_batch)
        else:
            original_gen = gen
            gen = itertools.islice(gen, max_loadable_frames)
    downscale_ratio = getattr(vae, "downscale_ratio", 8)
    frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
    if force_size != "Disabled" or vae is not None:
        new_size = target_size(width, height, force_size, custom_width, custom_height, downscale_ratio)
        if new_size[0] != width or new_size[1] != height:
            def rescale(frame):
                s = torch.from_numpy(np.fromiter(frame, np.dtype((np.float32, (height, width, 3)))))
                s = s.movedim(-1,1)
                s = common_upscale(s, new_size[0], new_size[1], "lanczos", "center")
                return s.movedim(1,-1).numpy()
            gen = itertools.chain.from_iterable(map(rescale, batched(gen, frames_per_batch)))
    else:
        new_size = width, height
    if vae is not None:
        gen = batched_vae_encode(gen, vae, frames_per_batch)
        vw,vh = new_size[0]//downscale_ratio, new_size[1]//downscale_ratio
        channels = getattr(vae, 'latent_channels', 4)
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (channels,vh,vw)))))
    else:
        #Some minor wizardry to eliminate a copy and reduce max memory by a factor of ~2
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (new_size[1], new_size[0], 3)))))
    if meta_batch is None and memory_limit is not None:
        try:
            next(original_gen)
            raise RuntimeError(f"Memory limit hit after loading {len(images)} frames. Stopping execution.")
        except StopIteration:
            pass
    if len(images) == 0:
        raise RuntimeError("No frames generated")

    #Setup lambda for lazy audio capture
    audio = lazy_get_audio(video, skip_first_frames * target_frame_time,
                               frame_load_cap*target_frame_time*select_every_nth)
    #Adjust target_frame_time for select_every_nth
    target_frame_time *= select_every_nth
    video_info = {
        "source_fps": fps,
        "source_frame_count": total_frames,
        "source_duration": duration,
        "source_width": width,
        "source_height": height,
        "loaded_fps": 1/target_frame_time,
        "loaded_frame_count": len(images),
        "loaded_duration": len(images) * target_frame_time,
        "loaded_width": new_size[0],
        "loaded_height": new_size[1],
    }
    if vae is None:
        return (images, len(images), audio, video_info, video)
    else:
        return ({"samples": images}, len(images), audio, video_info, video)



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
        return {"required": {
                    "video": (sorted(files),),
                     "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                     "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                     "custom_width": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                     "custom_height": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                     "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                     "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                     "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                     },
                "optional": {
                    "meta_batch": ("VHS_BatchManager",),
                    "vae": ("VAE",),
                },
                "hidden": {
                    "unique_id": "UNIQUE_ID"
                },
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = (imageOrLatent, "INT", "AUDIO", "VHS_VIDEOINFO", "STRING")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info", "video")

    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(strip_path(kwargs['video']))
        return load_video_cv(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return calculate_file_hash(image_path)

    @classmethod
    def VALIDATE_INPUTS(s, video, force_size, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True


class LoadVideoPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("STRING", {"placeholder": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                 "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                 "custom_width": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                 "custom_height": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": {
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
        return load_video_cv(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return validate_path(video, allow_none=True)
