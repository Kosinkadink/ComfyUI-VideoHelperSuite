import os
import itertools
import numpy as np
import torch
import subprocess
import re

import folder_paths
from comfy.utils import common_upscale
from .logger import logger
from .utils import BIGMAX, DIMMAX, calculate_file_hash, get_sorted_dir_files_from_directory, get_audio, lazy_eval, hash_path, validate_path, ffmpeg_path


video_extensions = ['webm', 'mp4', 'mkv', 'gif']


def is_gif(filename) -> bool:
    file_parts = filename.split('.')
    return len(file_parts) > 1 and file_parts[-1] == "gif"


def target_size(width, height, force_size, custom_width, custom_height) -> tuple[int, int]:
    if force_size == "Custom":
        return (custom_width, custom_height)
    elif force_size == "Custom Height":
        force_size = "?x"+str(custom_height)
    elif force_size == "Custom Width":
        force_size = str(custom_width)+"x?"

    if force_size != "Disabled":
        force_size = force_size.split("x")
        if force_size[0] == "?":
            width = (width*int(force_size[1]))//height
            #Limit to a multple of 8 for latent conversion
            width = int(width)+4 & ~7
            height = int(force_size[1])
        elif force_size[1] == "?":
            height = (height*int(force_size[0]))//width
            height = int(height)+4 & ~7
            width = int(force_size[0])
        else:
            width = int(force_size[0])
            height = int(force_size[1])
    return (width, height)

def ffmpeg_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                           select_every_nth, force_size, custom_width, custom_height,
                           batch_manager=None, unique_id=None):
    args_dummy = [ffmpeg_path, "-i", video, "-f", "null", "-"]
    size_base = None
    fps_base = None
    try:
        dummy_res =  subprocess.run(args_dummy, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE,check=True)
    except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the ffmepg subprocess:\n" \
                + e.stderr.decode("utf-8"))
    for line in dummy_res.stderr.decode("utf-8").split("\n"):
        match = re.search(", ([1-9]|\\d{2,})x(\\d+).*, ([\\d\\.]+) fps", line)
        if match is not None:
            size_base = [int(match.group(1)), int(match.group(2))]
            fps_base = float(match.group(3))
            alpha = re.search("(yuva|rgba)", line) is not None
            break
    else:
        raise Exception("Failed to parse video dimensions")

    args_all_frames = [ffmpeg_path, "-v", "error", "-an", "-i", video,
                         "-pix_fmt", "rgba" if alpha else "rgb24", "-fps_mode", "vfr"]

    if select_every_nth != 1:
        if force_rate == 0:
            force_rate = fps_base
        force_rate /= select_every_nth
    vfilters = []
    if force_rate != 0:
        vfilters.append("fps=fps="+str(force_rate) + ":round=up:start_time=0.001")
    if skip_first_frames > 0:
        vfilters.append(f"select=gt(n\\,{skip_first_frames-1})")
        vfilters.append("setpts=PTS-STARTPTS")
    if force_size != "Disabled":
        #TODO: let ffmpeg handle aspect ratio by setting unknown dimensions to -8?
        size = target_size(size_base[0], size_base[1], force_size, custom_width, custom_height)
        vfilters.append(f"scale={new_size[0]}:{new_size[1]}")
    else:
        size = size_base
    yield (size[0], size[1], fps_base, alpha)
    if len(vfilters) > 0:
        args_all_frames += ["-vf", ",".join(vfilters)]
    if frame_load_cap > 0:
        args_all_frames += ["-frames:v", str(frame_load_cap)]

    args_all_frames += ["-f", "rawvideo", "-"]
    try:
        with subprocess.Popen(args_all_frames, stdout=subprocess.PIPE) as proc:
            #Manually buffer enough bytes for an image
            bpi = size[0] * size[1] * (4 if alpha else 3)
            current_bytes = bytearray(bpi)
            current_offset=0
            prev_frame = None
            while True:
                bytes_read = proc.stdout.read(bpi - current_offset)
                if bytes_read is None:#sleep to wait for more data
                    time.sleep(.1)
                    continue
                if len(bytes_read) == 0:#EOF
                    break
                current_bytes[current_offset:len(bytes_read)] = bytes_read
                current_offset+=len(bytes_read)
                if current_offset == bpi:
                    if prev_frame is not None:
                        yield prev_frame
                    prev_frame = np.array(current_bytes, dtype=np.float32).reshape(size[1], size[0], 4 if alpha else 3) / 255.0
                    current_offset = 0
    except BrokenPipeError as e:
        raise Exception("An error occured in the ffmpeg subprocess:\n" \
                + proc.stderr.read().decode("utf-8"))
    if batch_manager is not None:
        batch_manager.inputs.pop(unique_id)
        batch_manager.has_closed_inputs = True
    if prev_frame is not None:
        yield prev_frame

def load_video_ffmpeg(video: str, force_rate: int, force_size: str,
                  custom_width: int,custom_height: int, frame_load_cap: int,
                  skip_first_frames: int, select_every_nth: int,
                  batch_manager=None, unique_id=None, prompt=None):

    if batch_manager is None or unique_id not in batch_manager.inputs:
        gen = ffmpeg_frame_generator(video, force_rate, frame_load_cap, skip_first_frames,
                                 select_every_nth, force_size, custom_width, custom_height,
                                    batch_manager, unique_id)
        (width, height, fps_base, has_alpha) = next(gen)
        if batch_manager is not None:
            batch_manager.inputs[unique_id] = (gen, width, height, fps_base, has_alpha)
    else:
        (gen, width, height, fps_base, has_alpha) = batch_manager.inputs[unique_id]
    if batch_manager is not None:
        gen = itertools.islice(gen, batch_manager.frames_per_batch)

    if has_alpha:
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 4)))))
        mask = 1 - images[:,:,:,3]
        images = images[:,:,:,:3]
    else:
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
        mask = torch.zeros((images.shape[0],1,1), dtype=torch.float32, device="cpu")


    if len(images) == 0:
        raise RuntimeError("No frames generated")

    #Setup lambda for lazy audio capture
    audio = lambda : get_audio(video, skip_first_frames / fps_base,
                               frame_load_cap / fps_base * select_every_nth)
    return (images, len(images), lazy_eval(audio), mask)


class LoadVideoUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
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
                    "batch_manager": ("VHS_BatchManager",)
                },
                "hidden": {
                    "prompt": "PROMPT",
                    "unique_id": "UNIQUE_ID"
                },
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("IMAGE", "INT", "VHS_AUDIO", "MASK")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "MASK")
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(kwargs['video'].strip("\""))
        return load_video_ffmpeg(**kwargs)

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
                "video": ("STRING", {"default": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
                 "force_size": (["Disabled", "Custom Height", "Custom Width", "Custom", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                 "custom_width": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                 "custom_height": ("INT", {"default": 512, "min": 0, "max": DIMMAX, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "batch_manager": ("VHS_BatchManager",)
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID"
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("IMAGE", "INT", "VHS_AUDIO", "MASK")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "MASK")
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        return load_video_ffmpeg(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        return validate_path(video, allow_none=True)
