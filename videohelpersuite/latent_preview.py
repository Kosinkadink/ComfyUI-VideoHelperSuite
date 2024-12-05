import asyncio
import subprocess
from PIL import Image

import latent_preview
import server
serv = server.PromptServer.instance

web = server.web

class PreviewInstance:
    def __init__(self):
        self.has_data = asyncio.Event()
        self.preview_at = 0
        self.data  = []
        self.closed = False

#TODO sid?
preview_instances = {}
def get_instance(node_id):
    if node_id not in preview_instances:
        preview_instances[node_id] = PreviewInstance()
    return preview_instances[node_id]

@serv.routes.get("/vhs/latentvideopreview")
async def latent_video_preview(request):
    query = request.rel_url.query
    if 'node_id' in query:
        instance = get_instance(query['node_id'])
    elif len(preview_instances) > 0:
        instance = next(preview_instances.values())
    else:
        return web.response(status=400)

    rate = 8.0
    args = ['ffmpeg','-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '512x512', '-r', str(rate), '-i', '-']
    args += ['-c:v', 'libvpx-vp9','-deadline', 'realtime', '-cpu-used', '8', '-f', 'webm', '-']
    try:
        proc = await asyncio.create_subprocess_exec(*args, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        try:
            resp = web.StreamResponse()
            resp.content_type = 'video/webm'
            await resp.prepare(request)
            async def read_loop():
                delay = asyncio.sleep(.1)
                while data := await proc.stdout.read(2**20):
                    await asyncio.gather(delay, resp.write(data))
                    delay = asyncio.sleep(.1)
            async def write_loop():
                await instance.has_data.wait()
                preview_at = 0
                delay = asyncio.create_task(asyncio.sleep(1/rate))
                frame_at = 0
                while True:
                    proc.stdin.write(instance.data[frame_at])
                    frame_at = (frame_at + 1) % len(instance.data)
                    if frame_at == instance.preview_at:
                        if not instance.has_data.is_set() and instance.closed:
                            proc.stdin.close()
                            delay.close()
                            return
                        await instance.has_data.wait()
                        instance.has_data.clear()
                    await asyncio.gather(proc.stdin.drain(), delay)
                    delay = asyncio.create_task(asyncio.sleep(1/rate))

            await asyncio.gather(read_loop(), write_loop())
            await proc.wait()
        except (ConnectionResetError, ConnectionError) as e:
            pass
        finally:
            print('ded')
            #Kill ffmpeg before the pipe is closed
            proc.kill()

    except BrokenPipeError as e:
        pass
    return resp

orig_get_previewer = latent_preview.get_previewer
def get_latent_video_previewer(device, latent_format):
    node_id = serv.last_node_id
    serv.send_sync('VHS_latentpreview', node_id)
    previewer = orig_get_previewer(device, latent_format)
    if not hasattr(previewer, "decode_latent_to_preview"):
        return None
    original_decode = previewer.decode_latent_to_preview
    def wrapped_decode(_, x0):
        inst = get_instance(node_id)
        num_images = x0.size(0)
        if len(inst.data) != num_images:
            inst.data = [b''] * num_images
        for i in range(num_images):
            sub_image = original_decode(x0[i:i+1])
            if sub_image.size[0] != 512 or sub_image.size[1] != 512:
                sub_image = sub_image.resize((512, 512),
                                             Image.Resampling.NEAREST)
            inst.data[i] = sub_image.tobytes()
        inst.has_data.set()
        return None
    previewer.decode_latent_to_preview_image = wrapped_decode
    return previewer
latent_preview.get_previewer = get_latent_video_previewer

