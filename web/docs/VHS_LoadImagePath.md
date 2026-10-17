# Load Image (Path) 🎥🅥🅗🅢

Load a single image from a given path

## Inputs

- **vae**: (optional) If provided the node will output latents instead of images.

## Outputs

- **IMAGE**: The loaded images
- **MASK**: The alpha channel of the loaded images.

## Widgets

- **image**: The image file to be loaded.
  - This is a VHS_PATH input. When edited, it provides a list of possible valid files or directories
  - <video src="https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/assets/4284322/729b7185-1fca-41d8-bc8d-a770bb2a5ce6" preload="none" muted loop controls></video>
  - The current top-most completion may be selected with Tab
  - You can navigate up a directory by pressing Ctrl+B (or Ctrl+W if supported by browser)
  - The filter on suggested file types can be disabled by pressing Ctrl+G.
  - If converted to an input, this functions as a string
- **force_size**: Allows for conveniently scaling the input without requiring an additional node. Provides options to maintain aspect ratio or conveniently target common training formats for Animate Diff
  - **custom_width**: Allows for an arbitrary width to be entered, cropping to maintain aspect ratio if both are set
  - **custom_height**: Allows for an arbitrary height to be entered, cropping to maintain aspect ratio if both are set
- **videopreview**: Displays a preview for the selected video input. Will only be shown if Advanced Previews is enabled. This preview will reflect the image_load_cap, skip_first_images, and select_every_nth values chosen. Additional preview options can be accessed with right click.
