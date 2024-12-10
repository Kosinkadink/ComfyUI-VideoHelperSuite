import asyncio
import subprocess
from PIL import Image
import time
import math
import io

import latent_preview
import server
serv = server.PromptServer.instance

orig_get_previewer = latent_preview.get_previewer
def get_latent_video_previewer(device, latent_format):
    node_id = serv.last_node_id
    previewer = orig_get_previewer(device, latent_format)
    if not hasattr(previewer, "decode_latent_to_preview"):
        return None
    first_preview = True
    serv.send_sync('VHS_latentpreview', node_id)
    last_time = time.time()
    c_index = 0
    original_decode = previewer.decode_latent_to_preview
    def wrapped_decode(_, x0):
        nonlocal first_preview, last_time, c_index
        #inst = get_instance(node_id)
        if x0.ndim == 5:
            #Keep batch major
            x0 = x0.movedim(2,1)
            x0 = x0.reshape((-1,)+x0.shape[-3:])
        num_images = x0.size(0)
        new_time = time.time()
        num_previews = min(math.ceil((new_time - last_time) * 8), num_images)
        if first_preview:
            first_preview = False
            serv.send_sync('VHS_latentpreview', num_images)
        last_time = new_time
        for _ in range(num_previews):
            sub_image = original_decode(x0[c_index:c_index+1])
            message = io.BytesIO()
            message.write(b'\x00\x00\x00\x01\x00\x00\x00\x01')
            message.write(c_index.to_bytes(length=4))
            if sub_image.size[0] != 512 or sub_image.size[1] != 512:
                sub_image = sub_image.resize((512, 512),
                                             Image.Resampling.NEAREST)
            sub_image.save(message, format="JPEG", quality=95, compress_level=1)
            serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE,
                           message.getvalue(), serv.client_id)
            c_index = (c_index +1) % num_images
        return None
    previewer.decode_latent_to_preview_image = wrapped_decode
    return previewer
latent_preview.get_previewer = get_latent_video_previewer


