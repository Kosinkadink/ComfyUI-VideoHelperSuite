# VAE Decode Batched 🎥🅥🅗🅢

Decode latents to images with a manually specified batch size

Some people have ran into VRAM issues when encoding or decoding large batches of images. As a workaround, this node lets you manually set a batch size when decoding latents.

Unless these issues have been encountered, it is simpler to use the native VAE Decode or to decode from a Video Combine directly

## Inputs

- **samples**: The latents to be decoded.
- **vae**: The VAE to use when decoding.

## Outputs

- **IMAGE**: The decoded images.

## Widgets

- **per_batch**: The maximum number of images to decode in each batch.
