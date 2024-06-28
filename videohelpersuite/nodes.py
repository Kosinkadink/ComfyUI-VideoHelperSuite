import os
import sys
import json
import subprocess
import numpy as np
import re
import datetime
from typing import List
import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from pathlib import Path
from string import Template
import itertools

import folder_paths
from .logger import logger
from .image_latent_nodes import *
from .load_video_nodes import LoadVideoUpload, LoadVideoPath
from .load_images_nodes import LoadImagesFromDirectoryUpload, LoadImagesFromDirectoryPath
from .batched_nodes import VAEEncodeBatched, VAEDecodeBatched
from .utils import ffmpeg_path, get_audio, hash_path, validate_path, requeue_workflow, gifski_path, calculate_file_hash, strip_path
from comfy.utils import ProgressBar

folder_paths.folder_names_and_paths["VHS_video_formats"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats"),
    ],
    [".json"]
)
audio_extensions = ['mp3', 'mp4', 'wav', 'ogg']

def gen_format_widgets(video_format):
    for k in video_format:
        if k.endswith("_pass"):
            for i in range(len(video_format[k])):
                if isinstance(video_format[k][i], list):
                    item = [video_format[k][i]]
                    yield item
                    video_format[k][i] = item[0]
        else:
            if isinstance(video_format[k], list):
                item = [video_format[k]]
                yield item
                video_format[k] = item[0]

def get_video_formats():
    formats = []
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_name = format_name[:-5]
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
        with open(video_format_path, 'r') as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format and gifski_path is None:
            #Skip format
            continue
        widgets = [w[0] for w in gen_format_widgets(video_format)]
        if (len(widgets) > 0):
            formats.append(["video/" + format_name, widgets])
        else:
            formats.append("video/" + format_name)
    return formats

def get_format_widget_defaults(format_name):
    video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    results = {}
    for w in gen_format_widgets(video_format):
        if len(w[0]) > 2 and 'default' in w[0][2]:
            default = w[0][2]['default']
        else:
            if type(w[0][1]) is list:
                default = w[0][1][0]
            else:
                #NOTE: This doesn't respect max/min, but should be good enough as a fallback to a fallback to a fallback
                default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[0][1]]
        results[w[0][0]] = default
    return results


def apply_format_widgets(format_name, kwargs):
    video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in gen_format_widgets(video_format):
        assert(w[0][0] in kwargs)
        if len(w[0]) > 3:
            w[0] = Template(w[0][3]).substitute(val=kwargs[w[0][0]])
        else:
            w[0] = str(kwargs[w[0][0]])
    return video_format

def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

def ffmpeg_process(args, video_format, video_metadata, file_path, env):

    res = None
    frame_data = yield
    total_frames_output = 0
    if video_format.get('save_metadata', 'False') != 'False':
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
        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now"]
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    #TODO: skip flush for increased speed
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                #Check if output file exists. If it does, the re-execution
                #will also fail. This obscures the cause of the error
                #and seems to never occur concurrent to the metadata issue
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                            + err.decode("utf-8"))
                #Res was not set
                print(err.decode("utf-8"), end="", file=sys.stderr)
                logger.warn("An error occurred when saving with metadata")
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                        + res.decode("utf-8"))
    yield total_frames_output
    if len(res) > 0:
        print(res.decode("utf-8"), end="", file=sys.stderr)

def gifski_process(args, video_format, file_path, env):
    frame_data = yield
    with subprocess.Popen(args + video_format['main_pass'] + ['-f', 'yuv4mpegpipe', '-'],
                          stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, env=env) as procff:
        with subprocess.Popen([gifski_path] + video_format['gifski_pass']
                              + ['-q', '-o', file_path, '-'], stderr=subprocess.PIPE,
                              stdin=procff.stdout, stdout=subprocess.PIPE,
                              env=env) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError as e:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception("An error occurred while creating gifski output\n" \
                        + "Make sure you are using gifski --version >=1.32.0\nffmpeg: " \
                        + resff.decode("utf-8") + '\ngifski: ' + resgs.decode("utf-8"))
    if len(resff) > 0:
        print(resff.decode("utf-8"), end="", file=sys.stderr)
    if len(resgs) > 0:
        print(resgs.decode("utf-8"), end="", file=sys.stderr)
    #should always be empty as the quiet flag is passed
    if len(outgs) > 0:
        print(outgs.decode("utf-8"))

def to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp)-2,0,-1):
        yield inp[i]

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats = get_video_formats()
        return {
            "required": {
                "frame_rate": (
                    "FLOAT",
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "images": ("IMAGE",),
                "audio": ("VHS_AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "latents": ("LATENT",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        vae=None
    ):
        if latents is not None:
            images = latents
        if images is None:
            return ((save_output, []),)
        if vae is not None:
            if isinstance(images, dict):
                images = images['samples']
            else:
                vae = None

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        if vae is not None:
            downscale_ratio = getattr(vae, "downscale_ratio", 8)
            width = images.size(3)*downscale_ratio
            height = images.size(2)*downscale_ratio
            frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
            #Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
            def batched(it, n):
                while batch := tuple(itertools.islice(it, n)):
                    yield batch
            def batched_encode(images, vae, frames_per_batch):
                for batch in batched(iter(images), frames_per_batch):
                    image_batch = torch.from_numpy(np.array(batch))
                    yield from vae.decode(image_batch)
            images = batched_encode(images, vae, frames_per_batch)
            first_image = next(images)
            #repush first_image
            images = itertools.chain([first_image], images)
        else:
            first_image = images[0]
            images = iter(images)
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # comfy counter workaround
            max_counter = 0

            # Loop through the existing files
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
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
            output_process = None

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, file)
        Image.fromarray(tensor_to_bytes(first_image)).save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")
        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                #Save timestamp information
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)
            frames = map(lambda x : Image.fromarray(tensor_to_bytes(x)), images)
            # Use pillow directly to save an animated image
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            #Acquire additional format_widget values
            kwargs = None
            if manual_format_widgets is None:
                if prompt is not None:
                    kwargs = prompt[unique_id]['inputs']
                else:
                    manual_format_widgets = {}
            if kwargs is None:
                kwargs = get_format_widget_defaults(format_ext)
                missing = {}
                for k in kwargs.keys():
                    if k in manual_format_widgets:
                        kwargs[k] = manual_format_widgets[k]
                    else:
                        missing[k] = kwargs[k]
                if len(missing) > 0:
                    logger.warn("Extra format values were not provided, the following defaults will be used: " + str(kwargs) + "\nThis is likely due to usage of ComfyUI-to-python. These values can be manually set by supplying a manual_format_widgets argument")

            video_format = apply_format_widgets(format_ext, kwargs)
            has_alpha = first_image.shape[-1] == 4
            dim_alignment = video_format.get("dim_alignment", 8)
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))
                images = map(pad, images)
                new_dims = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                            -first_image.shape[0] % dim_alignment + first_image.shape[0])
                dimensions = f"{new_dims[0]}x{new_dims[1]}"
                logger.warn("Output images were not of valid resolution and have had padding applied")
            else:
                dimensions = f"{first_image.shape[1]}x{first_image.shape[0]}"
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(num_frames)]
            else:
                loop_args = []
            if pingpong:
                if meta_batch is not None:
                    logger.error("pingpong is incompatible with batched output")
                images = to_pingpong(images)
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                if has_alpha:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                if has_alpha:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                    + loop_args

            images = map(lambda x: x.tobytes(), images)
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    #Performing a prepass requires keeping access to all frames.
                    #Potential solutions include keeping just output frames in
                    #memory or using 3 passes with intermediate file, but
                    #very long gifs probably shouldn't be encouraged
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                pre_pass_args = args[:13] + video_format['pre_pass']
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode("utf-8"))
            if "inputs_main_pass" in video_format:
                args = args[:13] + video_format['inputs_main_pass'] + args[13:]

            if output_process is None:
                if 'gifski_pass' in video_format:
                    output_process = gifski_process(args, video_format, file_path, env)
                else:
                    args += video_format['main_pass'] + bitrate_arg
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                #Proceed to first yield
                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image in images:
                pbar.update(1)
                output_process.send(image)
            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
            if meta_batch is None or meta_batch.has_closed_inputs:
                #Close pipe and wait for termination.
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                #batch is unfinished
                #TODO: Check if empty output breaks other custom nodes
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            output_files.append(file_path)

            if audio is not None and audio() is not False:
                # Create audio file if input was provided
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                if "audio_pass" not in video_format:
                    logger.warn("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]


                # FFmpeg command with audio re-encoding
                #TODO: expose audio quality options if format widgets makes it in
                #Reconsider forcing apad/shortest
                min_audio_dur = total_frames_output / frame_rate + 1
                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                            "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + ["-af", "apad=whole_dur="+str(min_audio_dur),
                               "-shortest", output_file_with_audio_path]

                try:
                    res = subprocess.run(mux_args, input=audio(), env=env,
                                         capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode("utf-8"))
                if res.stderr:
                    print(res.stderr.decode("utf-8"), end="", file=sys.stderr)
                output_files.append(output_file_with_audio_path)
                #Return this file with audio to the webui.
                #It will be muted unless opened or saved with right click
                file = output_file_with_audio

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
                "format": format,
                "frame_rate": frame_rate,
            }
        ]
        if num_frames == 1 and 'png' in format and '%03d' in file:
            previews[0]['format'] = 'image/png'
            previews[0]['filename'] = file.replace('%03d', '001')
        return {"ui": {"gifs": previews}, "result": ((save_output, output_files),)}
    @classmethod
    def VALIDATE_INPUTS(self, format, **kwargs):
        return True

class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        #Hide ffmpeg formats if ffmpeg isn't available
        return {
            "required": {
                "audio_file": ("STRING", {"default": "input/", "vhs_path_extensions": ['wav','mp3','ogg','m4a','flac']}),
                },
            "optional" : {"seek_seconds": ("FLOAT", {"default": 0, "min": 0})}
        }

    RETURN_TYPES = ("VHS_AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "load_audio"
    def load_audio(self, audio_file, seek_seconds):
        audio_file = strip_path(audio_file)
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        #Eagerly fetch the audio since the user must be using it if the
        #node executes, unlike Load Video
        audio = get_audio(audio_file, start_time=seek_seconds)
        return (lambda : audio,)

    @classmethod
    def IS_CHANGED(s, audio_file, seek_seconds):
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio_file, **kwargs):
        return validate_path(audio_file, allow_none=True)

class LoadAudioUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in audio_extensions):
                    files.append(f)
        return {"required": {
                    "audio": (sorted(files),),
                    "start_time": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                    "duration": ("FLOAT" , {"default": 0, "min": 0, "max": 10000000, "step": 0.01}),
                     },
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("VHS_AUDIO", )
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"

    def load_audio(self, start_time, duration, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(kwargs['audio']))
        if audio_file is None or validate_path(audio_file) != True:
            raise Exception("audio_file is not a valid path: " + audio_file)
        
        audio = get_audio(audio_file, start_time, duration)

        return (lambda : audio,)

    @classmethod
    def IS_CHANGED(s, audio, start_time, duration):
        audio_file = folder_paths.get_annotated_filepath(strip_path(audio))
        return hash_path(audio_file)

    @classmethod
    def VALIDATE_INPUTS(s, audio, **kwargs):
        audio_file = folder_paths.get_annotated_filepath(strip_path(audio))
        return validate_path(audio_file, allow_none=True)

class PruneOutputs:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "filenames": ("VHS_FILENAMES",),
                    "options": (["Intermediate", "Intermediate and Utility"],)
                    }
                }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "prune_outputs"

    def prune_outputs(self, filenames, options):
        if len(filenames[1]) == 0:
            return ()
        assert(len(filenames[1]) <= 3 and len(filenames[1]) >= 2)
        delete_list = []
        if options in ["Intermediate", "Intermediate and Utility", "All"]:
            delete_list += filenames[1][1:-1]
        if options in ["Intermediate and Utility", "All"]:
            delete_list.append(filenames[1][0])
        if options in ["All"]:
            delete_list.append(filenames[1][-1])

        output_dirs = [os.path.abspath("output"), os.path.abspath("temp")]
        for file in delete_list:
            #Check that path is actually an output directory
            if (os.path.commonpath([output_dirs[0], file]) != output_dirs[0]) \
                    and (os.path.commonpath([output_dirs[1], file]) != output_dirs[1]):
                        raise Exception("Tried to prune output from invalid directory: " + file)
            if os.path.exists(file):
                os.remove(file)
        return ()

class BatchManager:
    def __init__(self, frames_per_batch=-1):
        self.frames_per_batch = frames_per_batch
        self.inputs = {}
        self.outputs = {}
        self.unique_id = None
        self.has_closed_inputs = False
    def reset(self):
        self.close_inputs()
        for key in self.outputs:
            if getattr(self.outputs[key][-1], "gi_suspended", False):
                try:
                    self.outputs[key][-1].send(None)
                except StopIteration:
                    pass
        self.__init__(self.frames_per_batch)
    def has_open_inputs(self):
        return len(self.inputs) > 0
    def close_inputs(self):
        for key in self.inputs:
            if getattr(self.inputs[key][-1], "gi_suspended", False):
                try:
                    self.inputs[key][-1].send(1)
                except StopIteration:
                    pass
        self.inputs = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "frames_per_batch": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1})
                    },
                "hidden": {
                    "prompt": "PROMPT",
                    "unique_id": "UNIQUE_ID"
                },
                }

    RETURN_TYPES = ("VHS_BatchManager",)
    RETURN_NAMES = ("meta_batch",)
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "update_batch"

    def update_batch(self, frames_per_batch, prompt=None, unique_id=None):
        if unique_id is not None and prompt is not None:
            requeue = prompt[unique_id]['inputs'].get('requeue', 0)
        else:
            requeue = 0
        if requeue == 0:
            self.reset()
            self.frames_per_batch = frames_per_batch
            self.unique_id = unique_id
        #onExecuted seems to not be called unless some message is sent
        return (self,)


class VideoInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT", "FLOAT","INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "source_fps🟨",
        "source_frame_count🟨",
        "source_duration🟨",
        "source_width🟨",
        "source_height🟨",
        "loaded_fps🟦",
        "loaded_frame_count🟦",
        "loaded_duration🟦",
        "loaded_width🟦",
        "loaded_height🟦",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]
        
        source_info = []
        loaded_info = []

        for key in keys:
            source_info.append(video_info[f"source_{key}"])
            loaded_info.append(video_info[f"loaded_{key}"])

        return (*source_info, *loaded_info)


class VideoInfoSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT",)
    RETURN_NAMES = (
        "fps🟨",
        "frame_count🟨",
        "duration🟨",
        "width🟨",
        "height🟨",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]
        
        source_info = []

        for key in keys:
            source_info.append(video_info[f"source_{key}"])

        return (*source_info,)


class VideoInfoLoaded:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT",)
    RETURN_NAMES = (
        "fps🟦",
        "frame_count🟦",
        "duration🟦",
        "width🟦",
        "height🟦",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]
        
        loaded_info = []

        for key in keys:
            loaded_info.append(video_info[f"loaded_{key}"])

        return (*loaded_info,)



NODE_CLASS_MAPPINGS = {
    "VHS_VideoCombine": VideoCombine,
    "VHS_LoadVideo": LoadVideoUpload,
    "VHS_LoadVideoPath": LoadVideoPath,
    "VHS_LoadImages": LoadImagesFromDirectoryUpload,
    "VHS_LoadImagesPath": LoadImagesFromDirectoryPath,
    "VHS_LoadAudio": LoadAudio,
    "VHS_LoadAudioUpload": LoadAudioUpload,
    "VHS_PruneOutputs": PruneOutputs,
    "VHS_BatchManager": BatchManager,
    "VHS_VideoInfo": VideoInfo,
    "VHS_VideoInfoSource": VideoInfoSource,
    "VHS_VideoInfoLoaded": VideoInfoLoaded,
    # Latent and Image nodes
    "VHS_SplitLatents": SplitLatents,
    "VHS_SplitImages": SplitImages,
    "VHS_SplitMasks": SplitMasks,
    "VHS_MergeLatents": MergeLatents,
    "VHS_MergeImages": MergeImages,
    "VHS_MergeMasks": MergeMasks,
    "VHS_SelectEveryNthLatent": SelectEveryNthLatent,
    "VHS_SelectEveryNthImage": SelectEveryNthImage,
    "VHS_SelectEveryNthMask": SelectEveryNthMask,
    "VHS_GetLatentCount": GetLatentCount,
    "VHS_GetImageCount": GetImageCount,
    "VHS_GetMaskCount": GetMaskCount,
    "VHS_DuplicateLatents": DuplicateLatents,
    "VHS_DuplicateImages": DuplicateImages,
    "VHS_DuplicateMasks": DuplicateMasks,
    # Batched Nodes
    "VHS_VAEEncodeBatched": VAEEncodeBatched,
    "VHS_VAEDecodeBatched": VAEDecodeBatched,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoCombine": "Video Combine 🎥🅥🅗🅢",
    "VHS_LoadVideo": "Load Video (Upload) 🎥🅥🅗🅢",
    "VHS_LoadVideoPath": "Load Video (Path) 🎥🅥🅗🅢",
    "VHS_LoadImages": "Load Images (Upload) 🎥🅥🅗🅢",
    "VHS_LoadImagesPath": "Load Images (Path) 🎥🅥🅗🅢",
    "VHS_LoadAudio": "Load Audio (Path)🎥🅥🅗🅢",
    "VHS_LoadAudioUpload": "Load Audio (Upload)🎥🅥🅗🅢",
    "VHS_PruneOutputs": "Prune Outputs 🎥🅥🅗🅢",
    "VHS_BatchManager": "Meta Batch Manager 🎥🅥🅗🅢",
    "VHS_VideoInfo": "Video Info 🎥🅥🅗🅢",
    "VHS_VideoInfoSource": "Video Info (Source) 🎥🅥🅗🅢",
    "VHS_VideoInfoLoaded": "Video Info (Loaded) 🎥🅥🅗🅢",
    # Latent and Image nodes
    "VHS_SplitLatents": "Split Latent Batch 🎥🅥🅗🅢",
    "VHS_SplitImages": "Split Image Batch 🎥🅥🅗🅢",
    "VHS_SplitMasks": "Split Mask Batch 🎥🅥🅗🅢",
    "VHS_MergeLatents": "Merge Latent Batches 🎥🅥🅗🅢",
    "VHS_MergeImages": "Merge Image Batches 🎥🅥🅗🅢",
    "VHS_MergeMasks": "Merge Mask Batches 🎥🅥🅗🅢",
    "VHS_SelectEveryNthLatent": "Select Every Nth Latent 🎥🅥🅗🅢",
    "VHS_SelectEveryNthImage": "Select Every Nth Image 🎥🅥🅗🅢",
    "VHS_SelectEveryNthMask": "Select Every Nth Mask 🎥🅥🅗🅢",
    "VHS_GetLatentCount": "Get Latent Count 🎥🅥🅗🅢",
    "VHS_GetImageCount": "Get Image Count 🎥🅥🅗🅢",
    "VHS_GetMaskCount": "Get Mask Count 🎥🅥🅗🅢",
    "VHS_DuplicateLatents": "Duplicate Latent Batch 🎥🅥🅗🅢",
    "VHS_DuplicateImages": "Duplicate Image Batch 🎥🅥🅗🅢",
    "VHS_DuplicateMasks": "Duplicate Mask Batch 🎥🅥🅗🅢",
    # Batched Nodes
    "VHS_VAEEncodeBatched": "VAE Encode Batched 🎥🅥🅗🅢",
    "VHS_VAEDecodeBatched": "VAE Decode Batched 🎥🅥🅗🅢",
}
