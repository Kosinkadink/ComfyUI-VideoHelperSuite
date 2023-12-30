import os
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2

import folder_paths
from comfy.utils import common_upscale
from .logger import logger
from .utils import calculate_file_hash, get_sorted_dir_files_from_directory, get_audio, lazy_eval, hash_path, validate_path


video_extensions = ['webm', 'mp4', 'mkv', 'gif']


def is_gif(filename) -> bool:
    file_parts = filename.split('.')
    return len(file_parts) > 1 and file_parts[-1] == "gif"


def target_size(width, height, force_size) -> tuple[int, int]:
    if force_size != "Disabled":
        force_size = force_size.split("x")
        if force_size[0] == "?":
            width = (width*int(force_size[1]))//height
            #Limit to a multple of 8 for latent conversion
            #TODO: Consider instead cropping and centering to main aspect ratio
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

def load_video_cv(video: str, force_rate: int, force_size: str, frame_load_cap: int, skip_first_frames: int, select_every_nth: int):
    try:
        video_cap = cv2.VideoCapture(video)
        if not video_cap.isOpened():
            raise ValueError(f"{video} could not be loaded with cv.")
        # set video_cap to look at start_index frame
        images = []
        total_frame_count = 0
        total_frames_evaluated = -1
        frames_added = 0
        base_frame_time = 1/video_cap.get(cv2.CAP_PROP_FPS)
        width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if force_rate == 0:
            target_frame_time = base_frame_time
        else:
            target_frame_time = 1/force_rate
        time_offset=target_frame_time - base_frame_time
        while video_cap.isOpened():
            if time_offset < target_frame_time:
                is_returned, frame = video_cap.read()
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert frame to comfyui's expected format (taken from comfy's load image code)
            image = Image.fromarray(frame)
            image = ImageOps.exif_transpose(image)
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            images.append(image)
            frames_added += 1
            # if cap exists and we've reached it, stop processing frames
            if frame_load_cap > 0 and frames_added >= frame_load_cap:
                break
    finally:
        video_cap.release()
    if len(images) == 0:
        raise RuntimeError("No frames generated")
    images = torch.cat(images, dim=0)
    if force_size != "Disabled":
        new_size = target_size(width, height, force_size)
        if new_size[0] != width or new_size[1] != height:
            s = images.movedim(-1,1)
            s = common_upscale(s, new_size[0], new_size[1], "lanczos", "center")
            images = s.movedim(1,-1)
    # TODO: raise an error maybe if no frames were loaded?

    #Setup lambda for lazy audio capture
    audio = lambda : get_audio(video, skip_first_frames * target_frame_time,
                               frame_load_cap*target_frame_time)
    return (images, frames_added, lazy_eval(audio))


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
                     "force_rate": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                     "force_size": (["Disabled", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                     "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                     "skip_first_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
                     "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
                     },}

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("IMAGE", "INT", "VHS_AUDIO", )
    RETURN_NAMES = ("IMAGE", "frame_count", "audio",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(kwargs['video'].strip("\""))
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
                "video": ("STRING", {"default": "X://insert/path/here.mp4", "vhs_path_extensions": video_extensions}),
                "force_rate": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                "force_size": (["Disabled", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            },
        }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("IMAGE", "INT", "VHS_AUDIO", )
    RETURN_NAMES = ("IMAGE", "frame_count", "audio",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        if kwargs['video'] is None or validate_path(kwargs['video']) != True:
            raise Exception("video is not a valid path: " + kwargs['video'])
        return load_video_cv(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(s, video, force_size, **kwargs):
        return validate_path(video, allow_none=True)
