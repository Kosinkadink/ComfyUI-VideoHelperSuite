# Audio to legacy VHS_AUDIO 🎥🅥🅗🅢

utility function for compatibility with external nodes

VHS used to use an internal VHS_AUDIO format for routing audio between inputs and outputs. This format was intended to only be used internally and was designed with a focus on performance over ease of use. Since ComfyUI now has an internal AUDIO format, VHS now uses this format. However, some custom node packs were made that are external to both ComfyUI and VHS that use VHS_AUDIO. This node was added so that those external nodes can still function

## Inputs

- **audio**: An input in the standardized AUDIO format

## Outputs

- **vhs_audio**: An output in the legacy VHS_AUDIO format for use with external nodes
