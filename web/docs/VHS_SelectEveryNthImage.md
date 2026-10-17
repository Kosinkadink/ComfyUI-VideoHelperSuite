# Select Every Nth Image 🎥🅥🅗🅢

Keep only 1 image for every interval

## Inputs

- **images**: The input image

## Outputs

- **IMAGE**: The output images
- **count**: The number of images in the input

## Widgets

- **select_every_nth**: The interval from which one frame is kept. 1 means no frames are skipped.
- **skip_first_images**: A number of frames which that is skipped from the start. This applies before select_every_nth. As a result, multiple copies of the node can each have a different skip_first_frames to divide the image into groups
