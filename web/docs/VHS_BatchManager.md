# Meta Batch Manager 🎥🅥🅗🅢

Split the processing of a very long video into sets of smaller Meta Batches

The Meta Batch Manager allows for extremely long input videos to be processed when all other methods for fitting the content in RAM fail. It does not effect VRAM usage.

It must be connected to at least one Input (a Load Video or Load Images) AND at least one Video Combine

![](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/assets/4284322/7cb3fb7e-59d8-4cb2-a09f-9c6698de8b1f)

It functions by holding both the inputs and ouputs open between executions, and automatically requeue's the workflow until one of the inputs is unable to provide additional images.

Because each sub execution only contains a subset of the total frames, each sub execution creates a hard window which temporal smoothing can not be applied across. This results in jumps in the output.

## Outputs

- **meta_batch**: Add all connected nodes to this Meta Batch

## Widgets

- **frames_per_batch**: How many frames to process for each sub execution. If loading as image, each frame will use about 50MB of RAM (not VRAM), and this can safely be set in the 100-1000 range, depending on available memory. When loading and combining from latent space (no blue image noodles exist), this value can be much higher, around the 2,000 to 20,000 range
