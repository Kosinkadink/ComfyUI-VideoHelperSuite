import server
import folder_paths
import os
import subprocess
import threading

import asyncio
import av

from .utils import is_url, get_sorted_dir_files_from_directory, ffmpeg_path, \
        validate_sequence, is_safe_path, strip_path, try_download_video, ENCODE_ARGS
from comfy.k_diffusion.utils import FolderOfImages


web = server.web

@server.PromptServer.instance.routes.get("/vhs/viewvideo")
@server.PromptServer.instance.routes.get("/viewvideo")
async def pyav_view_video(request):
    query = request.rel_url.query
    path_res = await resolve_path(query)
    if isinstance(path_res, web.Response):
        return path_res
    file, filename, output_dir = path_res
    resp = web.StreamResponse()
    try:
        resp.content_type = 'video/webm'
        resp.headers["Content-Disposition"] = f"filename=\"{filename}\""
        await resp.prepare(request)
        loop = asyncio.get_event_loop()
        class BlockingFile:
            def __init__(self, async_file):
                self.lock = threading.Lock()
                self.async_file = async_file
                self.closed = False
                self.error = None
            def write(self, b):
                self.lock.acquire()
                if self.error:
                    raise self.error
                #NOTE assignment purely to stop premature garbage collection
                self.ctask = loop.call_soon_threadsafe(asyncio.create_task, self.do_write(b))
            def writable(self):
                return True
            def seekable(self):
                return False
            def close(self):
                if not self.closed:
                    self.lock.acquire()
                    self.closed = True
                if self.error:
                    raise self.error
            async def do_write(self, b):
                try:
                    await self.async_file.write(b)
                except Exception as e:
                    self.error = e
                finally:
                    self.lock.release()
        bresp = BlockingFile(resp)
        await asyncio.to_thread(pyav_transcode, query, file, bresp)
    except (ConnectionResetError, ConnectionError) as e:
        pass
    except Exception as e:
        raise
    return resp
def pyav_transcode(query, ifile, ofile):
    with av.open(ifile, 'r') as icont:
        #NOTE If ofile is 'closed' exiting the creates cascading errors that obfuscate cause
        #As ofile is only file like, failing to close it does not create a dangling reference
        ocont = av.open(ofile, 'w', format='matroska')
        istreams = {}
        processors = {}
        start_pts = 0
        if 'start_time' in query:
            #TODO Verify correctness of time base
            start_pts = int(float(query['start_time']), av.time_base)
            icont.seek(start_pts)
        if icont.streams.video:
            target_rate = query.get('frame_rate') or icont.streams.video[0].average_rate
            frame_load_cap = int(query['frame_load_cap']) if 'frame_load_cap' in query else float('inf')
            ostream = ocont.add_stream('libvpx-vp9', rate=target_rate)
            if 'deadline' in query:
                #Doesn't seem to function with pyav. Potentially placebo
                ostream.options['deadline'] = query['deadline']
            #TODO Calc appropriate value for cpu-used
            ostream.options['cpu-used'] = '8'
            istream = icont.streams.video[0]
            istreams['video'] = 0
            fg = av.filter.Graph()
            filters = []
            if 'force_size' in query:
                #TODO cropping?
                size = query['force_size'].split('x')
                if size[0] == '?':
                    size[0] = '-1'
                if size[1] == '?':
                    size[1] = '-1'
                filters.append(fg.add('scale', ':'.join(size)))
            #TODO: skip graph if no filters?
            fg.link_nodes(fg.add_buffer(template=icont.streams.video[0]),
                          *filters,
                          fg.add('buffersink')).configure()
            if 'skip_first_frames' in query:
                start_pts += int((int(query['skip_first_frames'])+1) /
                                 (istream.average_rate * istream.time_base))
                icont.seek(start_pts, stream=istream)
            cc = av.Codec('libvpx-vp9', 'r').create() if istream.codec_context.name == 'vp9' else istream
            def process_video(packet):
                nonlocal frame_load_cap
                for frame in cc.decode(packet):
                    frame_load_cap -= 1
                    if frame_load_cap <= 0:
                        return
                    if frame.pts < start_pts:
                        continue
                    #Should be required, but seems to cause increased stuttering with no upside
                    #frame.pts -= start_pts
                    fg.push(frame)
                    yield from ostream.encode(fg.pull())
            processors[icont.streams.video[0]] = process_video
        if icont.streams.audio:
            #TODO skip transcode if already desired codec
            astream = ocont.add_stream('libvorbis')
            istreams['audio'] = 0
            def process_audio(p):
                for frame in p.decode():
                    yield from astream.encode(frame)
            processors[icont.streams.audio[0]] = process_audio
        for packet in icont.demux(istreams):
            #TODO terminate encoding if out of frames?
            ocont.mux(processors[packet.stream](packet))
        #TODO: collate?
        for stream in ocont.streams.get():
            for packet in stream.encode(None):
                ocont.mux(packet)
        ocont.close()
    ofile.close()

query_cache = {}
@server.PromptServer.instance.routes.get("/vhs/queryvideo")
async def query_video(request):
    query = request.rel_url.query
    filepath = await resolve_path(query)
    if isinstance(filepath, web.Response):
        return filepath
    filepath = filepath[0]
    if filepath.endswith(".webp"):
        # ffmpeg doesn't support decoding animated WebP https://trac.ffmpeg.org/ticket/4907
        return web.json_response({})
    if filepath in query_cache and query_cache[filepath][0] == os.stat(filepath).st_mtime:
        source = query_cache[filepath][1]
    else:
        source = {}
        try:
            cont = av.open(filepath)
            stream = cont.streams.video[0]
            source['fps'] = float(stream.average_rate)
            source['duration'] = float(cont.duration * stream.time_base / 1000)

            cc = vpxcc if stream.codec_context.name == 'vp9' else stream
            def fit():
                for packet in cont.demux(video=0):
                    yield from cc.decode(packet)
            frame = next(fit())

            source['size'] = [frame.width, frame.height]
            source['alpha'] = 'a' in frame.format.name
            source['frames'] = stream.metadata.get('NUMBER_OF_FRAMES', round(source['duration'] * source['fps']))
            query_cache[filepath] = (os.stat(filepath).st_mtime, source)
        except Exception:
            pass
    if not 'frames' in source:
        return web.json_response({})
    loaded = {}
    loaded['duration'] = source['duration']
    loaded['duration'] -= float(query.get('start_time',0))
    loaded['fps'] = float(query.get('force_rate', 0)) or source['fps']
    loaded['duration'] -= int(query.get('skip_first_frames', 0)) / loaded['fps']
    loaded['fps'] /= int(query.get('select_every_nth', 1)) or 1
    loaded['frames'] = round(loaded['duration'] * loaded['fps'])
    return web.json_response({'source': source, 'loaded': loaded})

async def resolve_path(query):
    if "filename" not in query:
        return web.Response(status=204)
    filename = query["filename"]

    #Path code misformats urls on windows and must be skipped
    if is_url(filename):
        file = await asyncio.to_thread(try_download_video, filename) or file
        filname, output_dir = os.path.split(file)
        return file, filename, output_dir
    else:
        filename, output_dir = folder_paths.annotated_filepath(filename)

        type = query.get("type", "output")
        if type == "path":
            #special case for path_based nodes
            #NOTE: output_dir may be empty, but non-None
            output_dir, filename = os.path.split(strip_path(filename))
        if output_dir is None:
            output_dir = folder_paths.get_directory_by_type(type)

        if output_dir is None:
            return web.Response(status=204)

        if not is_safe_path(output_dir):
            return web.Response(status=204)

        if "subfolder" in query:
            output_dir = os.path.join(output_dir, query["subfolder"])

        filename = os.path.basename(filename)
        file = os.path.join(output_dir, filename)

        if query.get('format', 'video') == 'folder':
            if not os.path.isdir(file):
                return web.Response(status=204)
        else:
            if not os.path.isfile(file) and not validate_sequence(file):
                    return web.Response(status=204)
        return file, filename, output_dir

@server.PromptServer.instance.routes.get("/vhs/getpath")
@server.PromptServer.instance.routes.get("/getpath")
async def get_path(request):
    query = request.rel_url.query
    if "path" not in query:
        return web.Response(status=204)
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
    valid_items.sort(key=lambda f: os.stat(os.path.join(path,f)).st_mtime)
    return web.json_response(valid_items)
