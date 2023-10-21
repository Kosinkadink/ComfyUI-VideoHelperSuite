import os
import hashlib
import json
import subprocess
import shutil
import re
import time
import numpy as np
from typing import List
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import cv2
from pathlib import Path

import folder_paths
from comfy.k_diffusion.utils import FolderOfImages
from comfy.utils import common_upscale
from .logger import logger
from .image_latent_nodes import DuplicateImages, DuplicateLatents, GetImageCount, GetLatentCount, MergeImages, MergeLatents, SelectEveryNthImage, SelectEveryNthLatent, SplitLatents, SplitImages
from .utils import calculate_file_hash, get_sorted_dir_files_from_directory

folder_paths.folder_names_and_paths["video_formats"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats"),
    ],
    [".json"]
)

ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
    logger.info("ffmpeg could not be found. Using ffmpeg from imageio-ffmpeg.")
    from imageio_ffmpeg import get_ffmpeg_exe
    try:
        ffmpeg_path = get_ffmpeg_exe()
    except:
        logger.warning("ffmpeg could not be found. Outputs that require it have been disabled")


class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        if ffmpeg_path is not None:
            ffmpeg_formats = ["video/"+x[:-5] for x in folder_paths.get_filename_list("video_formats")]
        else:
            ffmpeg_formats = []
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_image": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("GIF",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def save_with_tempfile(self, args, metadata, file_path, frames, env):
        #Ensure temp directory exists
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)

        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        #metadata from file should  escape = ; # \ and newline
        #From my testing, though, only backslashes need escapes and = in particular causes problems
        #It is likely better to prioritize future compatibility with containers that don't support
        #or shouldn't use the comment tag for embedding metadata
        metadata = metadata.replace("\\","\\\\")
        metadata = metadata.replace(";","\\;")
        metadata = metadata.replace("#","\\#")
        #metadata = metadata.replace("=","\\=")
        metadata = metadata.replace("\n","\\\n")
        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write(metadata)
        args = args[:1] + ["-i", metadata_path] + args[1:] + [file_path]
        with subprocess.Popen(args, stdin=subprocess.PIPE, env=env) as proc:
            for frame in frames:
                proc.stdin.write(frame.tobytes())

    def combine_video(
        self,
        images,
        frame_rate: int,
        loop_count: int,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_image=True,
        prompt=None,
        extra_pnginfo=None,
    ):
        # convert images to numpy
        frames: List[Image.Image] = []
        for image in images:
            img = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            frames.append(img)

        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_image
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}_.png"
        file_path = os.path.join(full_output_folder, file)
        frames[0].save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        if pingpong:
            frames = frames + frames[-2:0:-1]

        format_type, format_ext = format.split("/")
        file = f"{filename}_{counter:05}_.{format_ext}"
        file_path = os.path.join(full_output_folder, file)
        if format_type == "image":
            # Use pillow directly to save an animated image
            frames[0].save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
            )
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                #Should never be reachable
                raise ProcessLookupError("Could not find ffmpeg")

            video_format_path = folder_paths.get_full_path("video_formats", format_ext + ".json")
            with open(video_format_path, 'r') as stream:
                video_format = json.load(stream)
            file = f"{filename}_{counter:05}_.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            dimensions = f"{frames[0].width}x{frames[0].height}"
            metadata_args = ["-metadata", "comment=" + json.dumps(video_metadata)]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                    + video_format['main_pass']
            # On linux, max arg length is Pagesize * 32 -> 131072
            # On windows, this around 32767 but seems to vary wildly by > 500
            # in a manor not solely related to other arguments
            if os.name == 'posix':
                max_arg_length = 4096*32
            else:
                max_arg_length = 32767 - len(" ".join(args + [metadata_args[0]] + [file_path])) - 1
            #test max limit
            #metadata_args[1] = metadata_args[1] + "a"*(max_arg_length - len(metadata_args[1])-1)

            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])
            if len(metadata_args[1]) >= max_arg_length:
                logger.info(f"Using fallback file for extremely long metadata: {len(metadata_args[1])}/{max_arg_length}")
                self.save_with_tempfile(args, metadata_args[1], file_path, frames, env)
            else:
                try:
                    with subprocess.Popen(args + metadata_args + [file_path],
                                          stdin=subprocess.PIPE, env=env) as proc:
                        for frame in frames:
                            proc.stdin.write(frame.tobytes())
                except FileNotFoundError as e:
                    if "winerror" in dir(e) and e.winerror == 206:
                        logger.warn("Metadata was too long. Retrying with fallback file")
                        self.save_with_tempfile(args, metadata_args[1], file_path, frames, env)
                    else:
                        raise
                except OSError as e:
                    if "errno" in dir(e) and e.errno == 7:
                        logger.warn("Metadata was too long. Retrying with fallback file")
                        self.save_with_tempfile(args, metadata_args[1], file_path, frames, env)
                    else:
                        raise

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_image else "temp",
                "format": format,
            }
        ]
        return {"ui": {"gifs": previews}}

class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        video_extensions = ['webm', 'mp4', 'mkv', 'gif']
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
                    files.append(f)
        return {"required": {
                    "video": (sorted(files), {"video_upload": True}),
                     "force_rate": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                     "force_size": (["Disabled", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                     "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                     "skip_first_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
                     #Consider adding start_frame/total_frames here?
                     #Might be a bit finicky since ffmpeg usually works in time/duration, not frame numbers
                     },}

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "frame_count",)
    FUNCTION = "load_video"

    def is_gif(self, filename):
        file_parts = filename.split('.')
        return len(file_parts) > 1 and file_parts[-1] == "gif"

    def target_size(self, width, height, force_size):
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
        return (width, height)

    def load_video_cv_fallback(self, video, force_rate, force_size, frame_load_cap, skip_first_frames):
        try:
            video_cap = cv2.VideoCapture(folder_paths.get_annotated_filepath(video))
            if not video_cap.isOpened():
                raise ValueError(f"{video} could not be loaded with cv fallback.")
            # set video_cap to look at start_index frame
            images = []
            total_frame_count = 0
            frames_added = 0
            base_frame_time = 1/video_cap.get(cv2.CAP_PROP_FPS)
            width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if force_rate == 0:
                target_frame_time = base_frame_time
            else:
                target_frame_time = 1/force_rate
            time_offset=0.0
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
                # TODO: do whatever operations need to happen, like force_size, etc

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
        images = torch.cat(images, dim=0)
        if force_size != "Disabled":
            new_size = self.target_size(width, height, force_size)
            if new_size[0] != width or new_size[1] != height:
                s = images.movedim(-1,1)
                s = common_upscale(s, new_size[0], new_size[1], "lanczos", "disabled")
                images = s.movedim(1,-1)
        # TODO: raise an error maybe if no frames were loaded?
        return (images, frames_added)

    def load_video(self, video, force_rate, force_size, frame_load_cap, skip_first_frames):
        # check if video is a gif - will need to use cv fallback to read frames
        # use cv fallback if ffmpeg not installed or gif
        if ffmpeg_path is None:
            return self.load_video_cv_fallback(video, force_rate, force_size, frame_load_cap, skip_first_frames)
        # otherwise, continue with ffmpeg
        video_path = folder_paths.get_annotated_filepath(video)
        args_dummy = [ffmpeg_path, "-i", video_path, "-f", "null", "-"]
        try:
            with subprocess.Popen(args_dummy, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as proc:
                for line in proc.stderr.readlines():
                    match = re.search(", ([1-9]|\\d{2,})x(\\d+)",line.decode('utf-8'))
                    if match is not None:
                        size = [int(match.group(1)), int(match.group(2))]
                        break
        except Exception as e:
            logger.info(f"Retrying with opencv due to ffmpeg error: {e}")
            return self.load_video_cv_fallback(video, force_rate, force_size, frame_load_cap, skip_first_frames)
        args_all_frames = [ffmpeg_path, "-i", video_path, "-v", "error",
                             "-pix_fmt", "rgb24"]

        vfilters = []
        if force_rate != 0:
            vfilters.append("fps="+str(force_rate))
        if skip_first_frames > 0:
            vfilters.append(f"select=gt(n\\,{skip_first_frames-1})")
        if frame_load_cap > 0:
            vfilters.append(f"select=gt({frame_load_cap}\\,n)")
        #manually calculate aspect ratio to ensure reads remain aligned
        if force_size != "Disabled":
            size = self.target_size(size[0], size[1], force_size)
            vfilters.append(f"scale={size[0]}:{size[1]}")
        if len(vfilters) > 0:
            args_all_frames += ["-vf", ",".join(vfilters)]

        args_all_frames += ["-f", "rawvideo", "-"]
        images = []
        try:
            with subprocess.Popen(args_all_frames, stdout=subprocess.PIPE) as proc:
                #Manually buffer enough bytes for an image
                bpi = size[0]*size[1]*3
                current_bytes = bytearray(bpi)
                current_offset=0
                while True:
                    bytes_read = proc.stdout.read(bpi - current_offset)
                    if bytes_read is None:#sleep to wait for more data
                        time.sleep(.2)
                        continue
                    if len(bytes_read) == 0:#EOF
                        break
                    current_bytes[current_offset:len(bytes_read)] = bytes_read
                    current_offset+=len(bytes_read)
                    if current_offset == bpi:
                        images.append(np.array(current_bytes, dtype=np.float32).reshape(size[1], size[0], 3) / 255.0)
                        current_offset = 0
        except Exception as e:
            logger.info(f"Retrying with opencv due to ffmpeg error: {e}")
            return self.load_video_cv_fallback(video, force_rate, force_size, frame_load_cap, skip_first_frames)

        images = torch.from_numpy(np.stack(images))
        return (images, images.size(0))

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid image file: {}".format(video)

        return True


class LoadImagesFromDirectory:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        directories = []
        for item in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, item)) and item != "clipspace":
                directories.append(item)
        return {
            "required": {
                "directory": (directories,),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "skip_first_images": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "load_images"

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    def load_images(self, directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1):
        directory = folder_paths.get_annotated_filepath(directory.strip())
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.")

        dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        images = []
        masks = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if limit_images and image_count >= image_load_cap:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_count += 1
        
        if len(images) == 0:
            raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

        return (torch.cat(images, dim=0), torch.stack(masks, dim=0), image_count)
    
    @classmethod
    def IS_CHANGED(s, directory: str, image_load_cap: int = 0, skip_first_images: int = 0, select_every_nth: int = 1):
        directory = folder_paths.get_annotated_filepath(directory.strip())
        if not os.path.isdir(directory):
            return False
        
        dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)
        dir_files = dir_files[:image_load_cap]

        m = hashlib.sha256()
        for filepath in dir_files:
            m.update(calculate_file_hash(filepath).encode()) # strings must be encoded before hashing
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, **kwargs):
        directory = folder_paths.get_annotated_filepath(directory.strip())
        if not os.path.isdir(directory):
            return f"Directory '{directory}' cannot be found."
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            return f"No files in directory '{directory}'."

        return True


NODE_CLASS_MAPPINGS = {
    "VHS_VideoCombine": VideoCombine,
    "VHS_LoadVideo": LoadVideo,
    "VHS_LoadImages": LoadImagesFromDirectory,
    # Latent and Image nodes
    "VHS_SplitLatents": SplitLatents,
    "VHS_SplitImages": SplitImages,
    "VHS_MergeLatents": MergeLatents,
    "VHS_MergeImages": MergeImages,
    "VHS_SelectEveryNthLatent": SelectEveryNthLatent,
    "VHS_SelectEveryNthImage": SelectEveryNthImage,
    "VHS_GetLatentCount": GetLatentCount,
    "VHS_GetImageCount": GetImageCount,
    "VHS_DuplicateLatents": DuplicateLatents,
    "VHS_DuplicateImages": DuplicateImages,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoCombine": "Video Combine 🎥🅥🅗🅢",
    "VHS_LoadVideo": "Load Video 🎥🅥🅗🅢",
    "VHS_LoadImages": "Load Images 🎥🅥🅗🅢",
    # Latent and Image nodes
    "VHS_SplitLatents": "Split Latent Batch 🎥🅥🅗🅢",
    "VHS_SplitImages": "Split Image Batch 🎥🅥🅗🅢",
    "VHS_MergeLatents": "Merge Latent Batches 🎥🅥🅗🅢",
    "VHS_MergeImages": "Merge Image Batches 🎥🅥🅗🅢",
    "VHS_SelectEveryNthLatent": "Select Every Nth Latent 🎥🅥🅗🅢",
    "VHS_SelectEveryNthImage": "Select Every Nth Image 🎥🅥🅗🅢",
    "VHS_GetLatentCount": "Get Latent Count 🎥🅥🅗🅢",
    "VHS_GetImageCount": "Get Image Count 🎥🅥🅗🅢",
    "VHS_DuplicateLatents": "Duplicate Latent Batch 🎥🅥🅗🅢",
    "VHS_DuplicateImages": "Duplicate Image Batch 🎥🅥🅗🅢",
}
