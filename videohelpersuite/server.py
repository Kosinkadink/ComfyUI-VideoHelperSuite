import server
import folder_paths
import os
import subprocess
import re

import asyncio

from .utils import is_url, get_sorted_dir_files_from_directory, ffmpeg_path, \
        validate_sequence, is_safe_path, strip_path, try_download_video, ENCODE_ARGS
from comfy.k_diffusion.utils import FolderOfImages


web = server.web

@server.PromptServer.instance.routes.get("/vhs/viewvideo")
@server.PromptServer.instance.routes.get("/viewvideo")
async def view_video(request):
    query = request.rel_url.query
    if "filename" not in query:
        return web.Response(status=404)
    filename = query["filename"]

    #Path code misformats urls on windows and must be skipped
    if is_url(filename):
        file = filename
        file = try_download_video(file) or file
    else:
        filename, output_dir = folder_paths.annotated_filepath(filename)

        type = request.rel_url.query.get("type", "output")
        if type == "path":
            #special case for path_based nodes
            #NOTE: output_dir may be empty, but non-None
            output_dir, filename = os.path.split(strip_path(filename))
        if output_dir is None:
            output_dir = folder_paths.get_directory_by_type(type)

        if output_dir is None:
            return web.Response(status=400)

        if not is_safe_path(output_dir):
            return web.Response(status=403)

        if "subfolder" in request.rel_url.query:
            output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])

        filename = os.path.basename(filename)
        file = os.path.join(output_dir, filename)

        if query.get('format', 'video') == 'folder':
            if not os.path.isdir(file):
                return web.Response(status=404)
        else:
            if not os.path.isfile(file) and not validate_sequence(file):
                    return web.Response(status=404)

    frame_rate = query.get('frame_rate', 8)
    if query.get('format', 'video') == "folder":
        #Check that folder contains some valid image file, get it's extension
        #ffmpeg seems to not support list globs, so support for mixed extensions seems unfeasible
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        concat_file = os.path.join(folder_paths.get_temp_directory(), "image_sequence_preview.txt")
        skip_first_images = int(query.get('skip_first_images', 0))
        select_every_nth = int(query.get('select_every_nth', 1))
        valid_images = get_sorted_dir_files_from_directory(file, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)
        if len(valid_images) == 0:
            return web.Response(status=400)
        with open(concat_file, "w") as f:
            f.write("ffconcat version 1.0\n")
            for path in valid_images:
                f.write("file '" + os.path.abspath(path) + "'\n")
                f.write("duration 0.125\n")
        in_args = ["-safe", "0", "-i", concat_file]
    else:
        in_args = ["-i", file]
        if '%' in file:
            in_args = ['-framerate', str(frame_rate)] + in_args
    #Do prepass to pull info
    #breaks skip_first frames if this default is ever actually needed
    base_fps = 30
    try:
        res = subprocess.run([ffmpeg_path] + in_args + ['-t', '0', '-f', 'null', '-'],
                             capture_output=True, check=True)
        match = re.search(': Video: (\\w+) .+, (\\d+) fps,', res.stderr.decode(*ENCODE_ARGS))
        if match:
            base_fps = float(match.group(2))
            if match.group(1) == 'vp9':
                #force libvpx for transparency
                in_args = ['-c:v', 'libvpx-vp9'] + in_args
    except subprocess.CalledProcessError as e:
        print("An error occurred in the ffmpeg prepass:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
    vfilters = []
    target_rate = float(query.get('force_rate', 0)) or base_fps
    modified_rate = target_rate / float(query.get('select_every_nth',1))
    start_time = 0
    if 'start_time' in query:
        start_time = float(query['start_time'])
    elif int(query.get('skip_first_frames', 0)) > 0:
        start_time = float(query.get('skip_first_frames'))/target_rate
        if start_time > 1/modified_rate:
            start_time += 1/modified_rate
    if start_time > 0:
        if start_time > 4:
            post_seek = ['-ss', '4']
            pre_seek = ['-ss', str(start_time - 4)]
        else:
            post_seek = ['-ss', str(start_time)]
            pre_seek = []
    else:
        pre_seek = []
        post_seek = []

    args = [ffmpeg_path, "-v", "error"] + pre_seek + in_args + post_seek
    if int(query.get('force_rate',0)) != 0:
        args += ['-r', str(modified_rate)]
    if query.get('force_size','Disabled') != "Disabled":
        size = query['force_size'].split('x')
        if size[0] == '?' or size[1] == '?':
            size[0] = "-2" if size[0] == '?' else f"'min({size[0]},iw)'"
            size[1] = "-2" if size[1] == '?' else f"'min({size[1]},ih)'"
        else:
            #Aspect ratio is likely changed. A more complex command is required
            #to crop the output to the new aspect ratio
            ar = float(size[0])/float(size[1])
            vfilters.append(f"crop=if(gt({ar}\\,a)\\,iw\\,ih*{ar}):if(gt({ar}\\,a)\\,iw/{ar}\\,ih)")
        size = ':'.join(size)
        vfilters.append(f"scale={size}")
    if len(vfilters) > 0:
        args += ["-vf", ",".join(vfilters)]
    if int(query.get('frame_load_cap', 0)) > 0:
        args += ["-frames:v", query['frame_load_cap']]
    #TODO:reconsider adding high frame cap/setting default frame cap on node

    args += ['-c:v', 'libvpx-vp9','-deadline', 'realtime', '-cpu-used', '8', '-f', 'webm', '-']

    try:
        with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
            try:
                resp = web.StreamResponse()
                resp.content_type = 'video/webm'
                resp.headers["Content-Disposition"] = f"filename=\"{filename}\""
                await resp.prepare(request)
                await asyncio.sleep(.1)
                while True:
                    bytes_read = proc.stdout.read(2**20)
                    delay = asyncio.create_task(asyncio.sleep(.1))
                    if bytes_read is None:
                        #TODO: check for timeout here
                        await delay
                        continue
                    if len(bytes_read) == 0:
                        break
                    await asyncio.gather(resp.write(bytes_read), delay)
                #Of dubious value given frequency of kill calls, but more correct
                proc.wait()
            except (ConnectionResetError, ConnectionError) as e:
                pass
            finally:
                #Kill ffmpeg before the pipe is closed
                proc.kill()

    except BrokenPipeError as e:
        pass
    return resp

@server.PromptServer.instance.routes.get("/vhs/getpath")
@server.PromptServer.instance.routes.get("/getpath")
async def get_path(request):
    query = request.rel_url.query
    if "path" not in query:
        return web.Response(status=404)
    #NOTE: path always ends in `/`, so this is functionally an lstrip
    path = os.path.abspath(strip_path(query["path"]))

    if not os.path.exists(path) or not is_safe_path(path):
        return web.json_response([])

    #Use get so None is default instead of keyerror
    valid_extensions = query.get("extensions")
    valid_items = []
    for item in os.scandir(path):
        try:
            if item.is_dir():
                valid_items.append(item.name + "/")
                continue
            if valid_extensions is None or item.name.split(".")[-1].lower() in valid_extensions:
                valid_items.append(item.name)
        except OSError:
            #Broken symlinks can throw a very unhelpful "Invalid argument"
            pass

    return web.json_response(valid_items)
