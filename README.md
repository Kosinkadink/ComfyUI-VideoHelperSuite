# ComfyUI-VideoHelperSuite
Nodes related to video workflows

## Nodes
### Load Video
Converts a video file into a series of images
- video: The video file to be loaded
- force_rate: Discards or duplicates frames as needed to hit a target frame rate. Disabled by setting to 0. This can be used to quickly match a suggested frame rate like the 8 fps of AnimateDiff.
- force_size: Allows for quick resizing to a number of suggested sizes. Several options allow you to set only width or height and determine the other from aspect ratio.
- frame_load_cap: The maximum number of frames which will be returned. This could also be thought of as the maximum batch size.
- skip_first_frames: How many frames to skip from the start of the video after adjusting for a forced frame rate. By incrementing this number by the frame_load_cap, you can easily process a longer input video in parts. 
- select_every_nth: Allows for skipping a number of frames without considering the base frame rate or risking frame duplication.  
A path variant of the Load Video node exists that allows loading videos from external paths, but does not currently support video previews
![step](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/assets/4284322/b5fc993c-5c9b-4608-afa4-48ae2e1380ef)
![resize](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/assets/4284322/98d2e78e-1c44-443c-a8fe-0dab0b5947f3)

### Load Image Sequence
Loads all image files from a subfolder. Options are similar to Load Video.
- image_load_cap: The maximum number of images which will be returned. This could also be thought of as the maximum batch size.
- skip_first_images: How many images to skip. By incrementing this number by image_load_cap, you can easily divide a long sequence of images into multiple batches.
- select_every_nth: Allows for skipping a number of images between every returned frame. As a set of image files doesn't have a frame rate, this is the only way to change the 'playback speed'

A path variant of Load Image sequence also exists.
### Video Combine
Combines a series of images into an output video
- frame_rate: How many of the input frames are displayed per second.  A higher frame rate means that the output video plays faster and has less duration. This should usually be kept to 8 for AnimateDiff, or matched to the force_rate of a Load Video node.
- loop_count: How many additional times the video should repeat
- filename_prefix: The base file name used for output.
  - You can save output to a subfolder: `subfolder/video`
  - Like the builtin Save Image node, you can add timestamps. `%date:yyyy-MM-ddThh:mm:ss%` might become 2023-10-31T6:45:25
- format: The file format to use. Advanced information on configuring or adding additional video formats can be found in the [Video Formats](#video-formats) section.
- pingpong: Causes the input to be played back in the reverse to create a clean loop.
- save_image: Whether the image should be put into the output directory or the temp directory
- crf: Describes the quality of the output video. A lower number gives a higher quality video and a larger file size, while a higher number gives a lower quality video with a smaller size. Scaling varies by codec, but visually lossless output generally occurs around 20.

## Video Previews
Load Video (Upload) and Video Combine provide animated previews, and this functionality may be added to additional nodes in the future.  
Nodes with previews provide additional functionality when right clicked
- Open preview
- Save preview
- Pause preview: Can improve performance with very large videos
- Hide preview: Can improve performance, save space, and also works with animated images
- Sync preview: Restarts all previews for side-by-side comparisons

## Video Formats
Those familiar with ffmpeg are able to add json files to the video_formats folders to add new output types to Video Combine. 
Consider the following example for av1-webm
```json
{
    "main_pass":
    [
        "-n", "-c:v", "libsvtav1",
        "-pix_fmt", "yuv420p10le"
    ],
     "extension": "webm",
     "environment": {"SVT_LOG": "1"}
}
```
Most configuration takes place in `main_pass`, which is a list of arguements that are passed to ffmpeg. 
- `"-n"` designates that the command should fail if a file of the same name already exists. This should never happen, but if some bug were to occur, it would ensure other files aren't overwritten.
- `"-c:v", "libsvtav1"` designates that the video should be encoded with an av1 codec using the new SVT-AV1 encoder. SVT-AV1 is much faster than libaom-av1, but may not exist in older versions of ffmpeg. Alternatively, av1_nvenc could be used for gpu encoding with newer nvidia cards. 
- `"-pix_fmt", "yuv420p10le"` designates the standard pixel format with 10-bit color. It's important that some pixel format be specified to ensure a nonconfigurable input pix_fmt isn't used. 


`extension` designates both the file extension and the container format that is used. If some of the above options are omitted from `main_pass` it can affect what default options are chosen.  
`environment` can optionally be provided to set environment variables during execution. For av1 it's used to reduce the verbosity of logging so that only major errors are displayed.
