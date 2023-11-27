import os
import sys
import json
import subprocess
import shutil
import numpy as np
import re
from typing import List
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pathlib import Path

import folder_paths
from .logger import logger
from .image_latent_nodes import DuplicateImages, DuplicateLatents, GetImageCount, GetLatentCount, MergeImages, MergeLatents, SelectEveryNthImage, SelectEveryNthLatent, SplitLatents, SplitImages
from .load_video_nodes import LoadVideoUpload, LoadVideoPath
from .load_images_nodes import LoadImagesFromDirectoryUpload, LoadImagesFromDirectoryPath

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

preferred_backend = "opencv"
if "VHS_PREFERRED_BACKEND" in os.environ:
    preferred_backend = os.environ['VHS_PREFERRED_BACKEND']

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
                "crf": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "save_metadata": ("BOOLEAN", {"default": True}),
                "audio_file": ("STRING", {"default": ""}),
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

    def combine_video(
        self,
        images,
        crf,
        frame_rate: int,
        loop_count: int,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_image=True,
        save_metadata=True,
        prompt=None,
        extra_pnginfo=None,
        audio_file=""
    ):
        # convert images to numpy
        images = images.cpu().numpy() * 255.0
        images = np.clip(images, 0, 255).astype(np.uint8)

        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_image
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
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

        # comfy counter workaround
        max_counter = 0

        # Loop through the existing files
        matcher = re.compile(f"{re.escape(filename)}_(\d+)_?\.[a-zA-Z0-9]+")
        for existing_file in os.listdir(full_output_folder):
            # Check if the file matches the expected format
            match = matcher.fullmatch(existing_file)
            if match:
                # Extract the numeric portion of the filename
                file_counter = int(match.group(1))
                # Update the maximum counter value if necessary
                if file_counter > max_counter:
                    max_counter = file_counter

        # Increment the counter by 1 to get the next available value
        counter = max_counter + 1

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, file)
        Image.fromarray(images[0]).save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        if pingpong:
            images = np.concatenate((images, images[-2:0:-1]))

        format_type, format_ext = format.split("/")
        file = f"{filename}_{counter:05}.{format_ext}"
        file_path = os.path.join(full_output_folder, file)
        if format_type == "image":
            frames = [Image.fromarray(f) for f in images]
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
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            dimensions = f"{len(images[0][0])}x{len(images[0])}"
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-", "-crf", str(crf) ] \
                    + video_format['main_pass']

            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])
            res = None
            if save_metadata:
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                metadata = json.dumps(video_metadata)
                metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
                #metadata from file should  escape = ; # \ and newline
                metadata = metadata.replace("\\","\\\\")
                metadata = metadata.replace(";","\\;")
                metadata = metadata.replace("#","\\#")
                metadata = metadata.replace("=","\\=")
                metadata = metadata.replace("\n","\\\n")
                metadata = "comment=" + metadata
                with open(metadata_path, "w") as f:
                    f.write(";FFMETADATA1\n")
                    f.write(metadata)
                m_args = args[:1] + ["-i", metadata_path] + args[1:]
                try:
                    res = subprocess.run(m_args + [file_path], input=images.tobytes(),
                                         capture_output=True, check=True, env=env)
                except subprocess.CalledProcessError as e:
                    #Res was not set
                    print(e.stderr.decode("utf-8"), end="", file=sys.stderr)
                    logger.warn("An error occurred when saving with metadata")

            if not res:
                try:
                    res = subprocess.run(args + [file_path], input=images.tobytes(),
                                         capture_output=True, check=True, env=env)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode("utf-8"))
            if res.stderr:
                print(res.stderr.decode("utf-8"), end="", file=sys.stderr)


            # Audio Injection ater video is created, saves additional video with -audio.mp4
            # Accepts mp3 and wav formats
            # TODO test unix and windows paths to make sure it works properly. Path module is Used

            audio_file_path = Path(audio_file)
            file_path = Path(file_path)

            # Check if 'audio_file' is not empty and the file exists
            if audio_file and audio_file_path.exists() and audio_file_path.suffix.lower() in ['.wav', '.mp3']:
                
                # Mapping of input extensions to output settings (extension, audio codec)
                format_settings = {
                    '.mov': ('.mov', 'pcm_s16le'),  # ProRes codec in .mov container
                    '.mp4': ('.mp4', 'aac'),        # H.264/H.265 in .mp4 container
                    '.mkv': ('.mkv', 'aac'),        # H.265 in .mkv container
                    '.webp': ('.webp', 'libvorbis'),
                    '.webm': ('.webm', 'libvorbis'),
                    '.av1': ('.webm', 'libvorbis')
                }

                output_extension, audio_codec = format_settings.get(file_path.suffix.lower(), (None, None))

                if output_extension and audio_codec:
                    # Modify output file name
                    output_file_with_audio_path = file_path.with_stem(file_path.stem + "-audio").with_suffix(output_extension)

                    # FFmpeg command with audio re-encoding
                    mux_args = [
                        ffmpeg_path, "-y", "-i", str(file_path), "-i", str(audio_file_path),
                        "-c:v", "copy", "-c:a", audio_codec, "-b:a", "192k", "-strict", "experimental", "-shortest", str(output_file_with_audio_path)
                    ]
                    
                    subprocess.run(mux_args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
                # Else block for unsupported video format can be added if necessar

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_image else "temp",
                "format": format,
            }
        ]
        return {"ui": {"gifs": previews}}


NODE_CLASS_MAPPINGS = {
    "VHS_VideoCombine": VideoCombine,
    "VHS_LoadVideo": LoadVideoUpload,
    "VHS_LoadVideoPath": LoadVideoPath,
    "VHS_LoadImages": LoadImagesFromDirectoryUpload,
    "VHS_LoadImagesPath": LoadImagesFromDirectoryPath,
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
    "VHS_LoadVideo": "Load Video (Upload) 🎥🅥🅗🅢",
    "VHS_LoadVideoPath": "Load Video (Path) 🎥🅥🅗🅢",
    "VHS_LoadImages": "Load Images (Upload) 🎥🅥🅗🅢",
    "VHS_LoadImagesPath": "Load Images (Path) 🎥🅥🅗🅢",
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
