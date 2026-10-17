# Video Info 🎥🅥🅗🅢

Splits information on a video into a numerous outputs

## Inputs

- **video_info**: A connection to a Load Video node

## Outputs

- **source_fps🟨**: The frame rate of the video
- **source_frame_count🟨**: How many total frames the video contains before accounting for frame rate or select_every_nth
- **source_duration🟨**: The length of images just returned in seconds
- **source_width🟨**: The width
- **source_height🟨**: The height
- **loaded_fps🟦**: The frame rate after accounting for force_rate and select_every_nth. This output is of particular use as it can be connected to the converted frame_rate input of a Video Combine node to ensure audio remains synchronized.
- **loaded_frame_count🟦**: The number of frames returned by the current execution. Identical to the frame_count returned by the node itself
- **loaded_duration🟦**: The duration in seconds of returned images after accounting for frame_load_cap
- **loaded_width🟦**: The width of the video after scaling. These coordinates are in image space even if loading to latent space
- **loaded_height🟦**: The height of the video after scaling. These coordinates are in image space even if loading to latent space
