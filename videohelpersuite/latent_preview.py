import asyncio
import subprocess
from PIL import Image
import time
import math
import io

import latent_preview
import server
serv = server.PromptServer.instance

from .utils import hook

class WrappedPreviewer(latent_preview.LatentPreviewer):
    def __init__(self, dltp, rate=8):
        self.first_preview = True
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        self.decode_latent_to_preview = dltp
    def decode_latent_to_preview_image(self, preview_format, x0):
        if x0.ndim == 5:
            #Keep batch major
            x0 = x0.movedim(2,1)
            x0 = x0.reshape((-1,)+x0.shape[-3:])
        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews/self.rate
        if num_previews > num_images:
            num_previews = num_images
        if self.first_preview:
            self.first_preview = False
            serv.send_sync('VHS_latentpreview', num_images)
        for _ in range(num_previews):
            sub_image = self.decode_latent_to_preview(x0[self.c_index:self.c_index+1])
            message = io.BytesIO()
            message.write((1).to_bytes(length=4)*2)
            message.write(self.c_index.to_bytes(length=4))
            if sub_image.size[0] > 512 or sub_image.size[1] > 512:
                if sub_image.size[0] > sub_image.size[1]:
                    resize = (512, int(sub_image.size[1]*512/sub_image.size[0]))
                else:
                    resize = (int(sub_image.size[0]*512/sub_image.size[1]), 512)
                sub_image = sub_image.resize(resize,
                                             Image.Resampling.NEAREST)
            sub_image.save(message, format="JPEG", quality=95, compress_level=1)
            serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE,
                           message.getvalue(), serv.client_id)
            self.c_index = (self.c_index + 1) % num_images
        return None

@hook(latent_preview, 'get_previewer')
def get_latent_video_previewer(device, latent_format, *args, **kwargs):
    node_id = serv.last_node_id
    previewer = get_latent_video_previewer.__wrapped__(device, latent_format, *args, **kwargs)
    try:
        prev_setting = next(serv.prompt_queue.currently_running.values().__iter__())[3] \
                ['extra_pnginfo']['workflow']['extra'].get('VHS_latentpreview', False)
        rate_setting = next(serv.prompt_queue.currently_running.values().__iter__())[3] \
                ['extra_pnginfo']['workflow']['extra'].get('VHS_latentpreviewrate', 8)
    except:
        #For safety since there's lots of keys, any of which can fail
        prev_setting = False
    if not prev_setting or not hasattr(previewer, "decode_latent_to_preview"):
        return previewer
    return WrappedPreviewer(previewer.decode_latent_to_preview, rate_setting)
