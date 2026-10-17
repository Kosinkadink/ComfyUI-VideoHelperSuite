# Video Info Source 🎥🅥🅗🅢

Splits information on a video into a numerous outputs describing the file itself without accounting for load options

## Inputs

- **video_info**: A connection to a Load Video node

## Outputs

- **source_fps🟨**: The frame rate of the video
- **source_frame_count🟨**: How many total frames the video contains before accounting for frame rate or select_every_nth
- **source_duration🟨**: The length of images just returned in seconds
- **source_width🟨**: The original width
- **source_height🟨**: The original height
