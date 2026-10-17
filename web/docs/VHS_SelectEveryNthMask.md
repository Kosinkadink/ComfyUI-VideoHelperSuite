# Select Every Nth Mask 🎥🅥🅗🅢

Keep only 1 mask for every interval

## Inputs

- **mask**: The input mask

## Outputs

- **MASK**: The output mask
- **count**: The number of mask in the input

## Widgets

- **select_every_nth**: The interval from which one frame is kept. 1 means no frames are skipped.
- **skip_first_mask**: A number of frames which that is skipped from the start. This applies before select_every_nth. As a result, multiple copies of the node can each have a different skip_first_frames to divide the mask into groups
