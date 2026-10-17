# Select Every Nth Latent 🎥🅥🅗🅢

Keep only 1 latent for every interval

## Inputs

- **latents**: The input latent

## Outputs

- **LATENT**: The output latents
- **count**: The number of latents in the input

## Widgets

- **select_every_nth**: The interval from which one frame is kept. 1 means no frames are skipped.
- **skip_first_latents**: A number of frames which that is skipped from the start. This applies before select_every_nth. As a result, multiple copies of the node can each have a different skip_first_frames to divide the latent into groups
