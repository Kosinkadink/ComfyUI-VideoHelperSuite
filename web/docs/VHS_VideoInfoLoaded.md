# Video Info Loaded 🎥🅥🅗🅢

Splits information on a video into a numerous outputs describing the file itself after accounting for load options

## Inputs

- **video_info**: A connection to a Load Video node

## Outputs

- **loaded_fps🟦**: The frame rate after accounting for force_rate and select_every_nth. This output is of particular use as it can be connected to the converted frame_rate input of a Video Combine node to ensure audio remains synchronized.
- **loaded_frame_count🟦**: The number of frames returned by the current execution. Identical to the frame_count returned by the node itself
- **loaded_duration🟦**: The duration in seconds of returned images after accounting for frame_load_cap
- **loaded_width🟦**: The width of the video after scaling. This is the dimension of the corresponding image even if loading as a latent directly
- **loaded_height🟦**: The height of the video after scaling. This is the dimension of the corresponding image even if loading as a latent directly
