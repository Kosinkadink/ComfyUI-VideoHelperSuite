# VAE Encode Batched 🎥🅥🅗🅢

Encode images as latents with a manually specified batch size.

Some people have ran into VRAM issues when encoding or decoding large batches of images. As a workaround, this node lets you manually set a batch size when encoding images.

Unless these issues have been encountered, it is simpler to use the native VAE Encode or to encode directly from a Load Video

## Inputs

- **pixels**: The images to be encoded.
- **vae**: The VAE to use when encoding.

## Outputs

- **LATENT**: The encoded latents.

## Widgets

- **per_batch**: The maximum number of images to encode in each batch.
