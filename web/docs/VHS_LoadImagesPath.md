# Load Images (Path) 🎥🅥🅗🅢

Loads a sequence of images from an arbitrary path

## Inputs

- **meta_batch**: (optional) Connect to a Meta Batch manager to divide extremely long sequences into sub batches. See the documentation for Meta Batch Manager

## Outputs

- **IMAGE**: The loaded images
- **MASK**: The alpha channel of the loaded images.
- **frame_count**: The length of images just returned

## Widgets

- **directory**: The directory images will be loaded from. Filtered to process jpg, png, ppm, bmp, tif, and webp files
  - This is a VHS_PATH input. When edited, it provides a list of possible valid files or directories
  - <video src="https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/assets/4284322/729b7185-1fca-41d8-bc8d-a770bb2a5ce6" preload="none" muted loop controls></video>
  - The current top-most completion may be selected with Tab
  - You can navigate up a directory by pressing Ctrl+B (or Ctrl+W if supported by browser)
  - The filter on suggested file types can be disabled by pressing Ctrl+G.
  - If converted to an input, this functions as a string
- **image_load_cap**: The maximum number of images to load. If 0, all images are loaded.
- **skip_first_images**: A number of images which are discarded before producing output.
- **select_every_nth**: Keeps only the first of every n frames and discard the rest.
- **videopreview**: Displays a preview for the selected video input. Will only be shown if Advanced Previews is enabled. This preview will reflect the image_load_cap, skip_first_images, and select_every_nth values chosen. Additional preview options can be accessed with right click.
