import os
import json
import subprocess
import shutil
import re
import time
import numpy as np
from typing import List
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from .logger import logger

ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path is None:
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
                    {"default": 8, "min": 1, "max": 24, "step": 1},
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
    CATEGORY = "Video Helper Suite"
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
            ffmpeg_path = shutil.which("ffmpeg")
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
        video_extensions = ['webm', 'mp4', 'mkv']
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
                    files.append(f)
        return {"required":
                    {"video": (sorted(files), {"video_upload": True}),
                     "force_rate": ("INT", {"default": 0, "min": 0, "max": 24, "step": 1}),
                     "force_size": (["Disabled", "256x?", "?x256", "256x256", "512x?", "?x512", "512x512"],),
                     #Consider adding start_frame/total_frames here?
                     #Might be a bit finicky since ffmpeg usually works in time/duration, not frame numbers
                     },}

    CATEGORY = "Video Helper Suite"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_video"
    def load_video(self, video, force_rate, force_size):
        video_path = folder_paths.get_annotated_filepath(video)
        args_dummy = ["ffmpeg", "-i", video_path, "-f", "null", "-"]
        with subprocess.Popen(args_dummy, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as proc:
            for line in proc.stderr.readlines():
                match = re.search(", ([1-9]|\\d{2,})x(\\d+),",line.decode('ascii'))
                if match is not None:
                    size = [int(match.group(1)), int(match.group(2))]
                    break
        args_all_frames = ["ffmpeg", "-i", video_path, "-v", "error",
                             "-pix_fmt", "rgb24"]

        if force_rate != 0:
            args_all_frames += ["-r", str(force_rate)]
        #manually calculate aspect ratio to ensure reads remain aligned
        if force_size != "Disabled":
            force_size = force_size.split("x")
            if force_size[0] == "?":
                size[0] = (size[0]*int(force_size[1]))//size[1]
                size[1] = int(force_size[1])
            elif force_size[1] == "?":
                size[1] = (size[1]*int(force_size[0]))//size[0]
                size[0] = int(force_size[0])
            args_all_frames += ["-vf", f"scale={size[0]}:{size[1]}"]

        args_all_frames += ["-f", "rawvideo", "-"]
        images = []
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
            if current_offset != 0:
                logger.warn(f'{current_offset} bytes left over when loading image')

        images = torch.from_numpy(np.stack(images))
        return (images,)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, video, force_rate, force_size):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid image file: {}".format(video)

        return True

NODE_CLASS_MAPPINGS = {
    "VHS_VideoCombine": VideoCombine,
    "VHS_LoadVideo": LoadVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoCombine": "Video Combine",
    "VHS_LoadVideo": "Load Video",
}
