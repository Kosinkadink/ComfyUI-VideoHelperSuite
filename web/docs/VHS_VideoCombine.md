# Video Combine 🎥🅥🅗🅢

Combine an image sequence into a video

## Inputs

- **images**: The images to be turned into a video
- **audio**: (optional) audio to add to the video
- **meta_batch**: (optional) Connect to a Meta Batch manager to divide extremely long image sequences into sub batches. See the documentation for Meta Batch Manager
- **vae**: (optional) If provided, the node will take latents as input instead of images. This drastically reduces the required RAM (not VRAM) when working with long (100+ frames) sequences
  - Unlike on Load Video, this isn't always a strict upgrade over using a standalone VAE Decode.
  - If you have multiple Video Combine outputs, then the VAE decode will be performed for each output node increasing execution time
  - If you make any change to output settings on the Video Combine (such as changing the output format), the VAE decode will be performed again as the decoded result is (by design) not cached

## Widgets

- **frame_rate**: The frame rate which will be used for the output video. Consider converting this to an input and connecting this to a Load Video with Video Info(Loaded)->fps. When including audio, failure to properly set this will result in audio desync
- **loop_count**: The number of additional times the video should repeat. Can cause performance issues when used with long (100+ frames) sequences
- **filename_prefix**: A prefix to add to the name of the output filename. This can include subfolders or format strings.
- **format**: The output format to use. Formats starting with, 'image' are saved with PIL, but formats starting with 'video' utilize the video_formats system. 'video' options require ffmpeg and selecting one frequently adds additional options to the node.
- **pingpong**: Play the video normally, then repeat the video in reverse so that it 'pingpongs' back and forth. This is frequently used to minimize the appearance of skips on very short animations.
- **save_output**: Specifies if output files should be saved to the output folder, or the temporary output folder
- **videopreview**: Displays a preview for the processed result. If advanced previews is enabled, the output is always converted to a format viewable from the browser. If the video has audio, it will also be previewed when moused over. Additional preview options can be accessed with right click.

## Common Format Widgets

- **crf**: Determines how much to prioritize quality over filesize. Numbers vary between formats, but on each format that includes it, the default value provides visually loss less output
- **pix_fmt**: The pixel format to use for output. Alternative options will often have higher quality at the cost of increased file size and reduced compatibility with external software.
  - **yuv420p**: The most common and default format
  - **yuv420p10le**: Use 10 bit color depth. This can improve color quality when combined with 16bit input color depth
  - **yuva420p**: Include transparency in the output video
- **input_color_depth**: VHS supports outputting 16bit images. While this produces higher quality output, the difference usually isn't visible without postprocessing and it significantly increases file size and processing time.
- **save_metadata**: Determines if metadata for the workflow should be included in the output video file
