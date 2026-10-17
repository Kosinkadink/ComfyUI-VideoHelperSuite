# Load Video 🎥🅥🅗🅢

Loads a video from the input folder

## Inputs

- **meta_batch**: (optional) Connect to a Meta Batch manager to divide extremely long sequences into sub batches. See the documentation for Meta Batch Manager
- **vae**: (optional) If provided the node will output latents instead of images. This drastically reduces the required RAM (not VRAM) when working with long (100+ frames) sequences
  - Using this is strongly encouraged unless connecting to a node that requires a blue image connection such as Apply Controllnet

## Outputs

- **IMAGE**: The loaded images
- **frame_count**: The length of images just returned
- **audio**: The audio from the loaded video
- **video_info**: Exposes additional info about the video such as the source frame rate, or the total length
- **LATENT**: The loaded images pre-converted to latents. Only available when a vae is connected

## Widgets

- **video**: The video file to be loaded. Lists all files with a video extension in the ComfyUI/Input folder
- **force_rate**: Drops or duplicates frames so that the produced output has the target frame rate. Many motion models are trained on videos of a specific frame rate and will give better results if input matches that frame rate. If set to 0, all frames are returned. May give unusual results with inputs that have a variable frame rate like animated gifs. Reducing this value can also greatly reduce the execution time and memory requirements.
- **force_size**: Previously was used to provide suggested resolutions. Instead, custom_width and custom_height can be disabled by setting to 0.
- **custom_width**: Allows for an arbitrary width to be entered, cropping to maintain aspect ratio if both are set
- **custom_height**: Allows for an arbitrary height to be entered, cropping to maintain aspect ratio if both are set
- **frame_load_cap**: The maximum number of frames to load. If 0, all frames are loaded.
- **skip_first_frames**: A number of frames which are discarded before producing output.
- **select_every_nth**: Similar to frame rate. Keeps only the first of every n frames and discard the rest. Has better compatibility with variable frame rate inputs such as gifs. When combined with force_rate, select_every_nth_applies after force_rate so the resulting output has a frame rate equivalent to force_rate/select_every_nth. select_every_nth does not apply to skip_first_frames
- **format**: Updates other widgets so that only values supported by the given format can be entered and provides recommended defaults.
- **choose video to upload**: An upload button is provided to upload local files to the input folder
- **videopreview**: Displays a preview for the selected video input. If advanced previews is enabled, this preview will reflect the frame_load_cap, force_rate, skip_first_frames, and select_every_nth values chosen. If the video has audio, it will also be previewed when moused over. Additional preview options can be accessed with right click.
