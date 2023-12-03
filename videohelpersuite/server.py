import server
import folder_paths
import os
import subprocess
from .utils import is_url

web = server.web

def is_safe(path):
    if "VHS_UNSAFE_PATHS" in os.environ:
        return True
    basedir = os.path.abspath('.')
    try:
        common_path = os.path.commonpath([basedir, path])
    except:
        #Different drive on windows
        return False
    return common_path == basedir

@server.PromptServer.instance.routes.get("/viewvideo")
async def view_video(request):
    query = request.rel_url.query
    if "filename" not in query:
        return web.Response(status=404)
    filename = query["filename"]

    #Path code misformats urls on windows and must be skipped
    if is_url(filename):
        file = filename
    else:
        filename, output_dir = folder_paths.annotated_filepath(filename)

        type = request.rel_url.query.get("type", "output")
        if type == "path":
            #special case for path_based nodes
            #NOTE: output_dir may be empty, but non-None
            output_dir, filename = os.path.split(filename)
        if output_dir is None:
            output_dir = folder_paths.get_directory_by_type(type)

        if output_dir is None:
            return web.Response(status=400)

        if not is_safe(output_dir):
            return web.Response(status=403)

        if "subfolder" in request.rel_url.query:
            output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])

        filename = os.path.basename(filename)
        file = os.path.join(output_dir, filename)

        if not os.path.isfile(file):
            return web.Response(status=404)

    args = ["ffmpeg", "-v", "error","-an", "-i", file]
    vfilters = []
    if int(query.get('force_rate',0)) != 0:
        vfilters.append("fps=fps="+query['force_rate'] + ":round=up:start_time=0.001")
    if int(query.get('skip_first_frames', 0)) > 0:
        vfilters.append(f"select=gt(n\\,{int(query['skip_first_frames'])-1})")
    if int(query.get('select_every_nth', 1)) > 1:
        vfilters.append(f"select=not(mod(n\\,{query['select_every_nth']}))")
    if query.get('force_size','Disabled') != "Disabled":
        size = query['force_size'].split('x')
        size[0] = "-2" if size[0] == '?' else f"'min({size[0]},iw)'"
        size[1] = "-2" if size[1] == '?' else f"'min({size[1]},ih)'"
        size = ':'.join(size)
        vfilters.append(f"scale={size}")
    vfilters.append("setpts=PTS-STARTPTS")
    if len(vfilters) > 0:
        args += ["-vf", ",".join(vfilters)]
    if int(query.get('frame_load_cap', 0)) > 0:
        args += ["-frames:v", query['frame_load_cap']]
    args += ['-c:v', 'libvpx-vp9','-deadline', 'realtime', '-cpu-used', '8', '-f', 'webm', '-']

    try:
        with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
            try:
                resp = web.StreamResponse()
                resp.content_type = 'video/webm'
                resp.headers["Content-Disposition"] = f"filename=\"{filename}\""
                await resp.prepare(request)
                while True:
                    bytes_read = proc.stdout.read()
                    if bytes_read is None:
                        #TODO: check for timeout here
                        time.sleep(.1)
                        continue
                    if len(bytes_read) == 0:
                        break
                    await resp.write(bytes_read)
            except ConnectionResetError as e:
                #Kill ffmpeg before stdout closes
                proc.kill()
    except BrokenPipeError as e:
        pass
    return resp

@server.PromptServer.instance.routes.get("/getpath")
async def get_path(request):
    query = request.rel_url.query
    if "path" not in query:
        return web.Response(status=404)
    path = os.path.abspath(query["path"])

    if not os.path.exists(path) or not is_safe(path):
        return web.json_response([])

    #Use get so None is default instead of keyerror
    valid_extensions = query.get("extensions")
    valid_items = []
    for item in os.scandir(path):
        if item.is_dir():
            valid_items.append(item.name + "/")
            continue
        if valid_extensions is None or item.name.split(".")[-1] in valid_extensions:
            valid_items.append(item.name)

    return web.json_response(valid_items)
