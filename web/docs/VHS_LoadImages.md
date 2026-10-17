# Load Images 🎥🅥🅗🅢

Loads a sequence of images from a subdirectory of the input folder

## Inputs

- **meta_batch**: (optional) Connect to a Meta Batch manager to divide extremely long sequences into sub batches. See the documentation for Meta Batch Manager

## Outputs

- **IMAGE**: The loaded images
- **MASK**: The alpha channel of the loaded images.
- **frame_count**: The length of images just returned

## Widgets

- **directory**: The directory images will be loaded from. Filtered to process jpg, png, ppm, bmp, tif, and webp files
- **image_load_cap**: The maximum number of images to load. If 0, all images are loaded.
- **start_time**: A timestamp, in seconds from the start of the video, to start loading frames from. 
- **choose folder to upload**: An upload button is provided to upload a local folder containing images to the input folder
- **videopreview**: Displays a preview for the selected video input. Will only be shown if Advanced Previews is enabled. This preview will reflect the image_load_cap, skip_first_images, and select_every_nth values chosen. Additional preview options can be accessed with right click.
