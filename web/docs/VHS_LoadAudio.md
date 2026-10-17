# Load Audio (Path) 🎥🅥🅗🅢

Loads an audio file from an arbitrary path

## Outputs

- **audio**: The loaded audio

## Widgets

- **audio_file**: The audio file to be loaded.
  - This is a VHS_PATH input. When edited, it provides a list of possible valid files or directories
  - <video src="https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/assets/4284322/729b7185-1fca-41d8-bc8d-a770bb2a5ce6" preload="none" muted loop controls></video>
  - The current top-most completion may be selected with Tab
  - You can navigate up a directory by pressing Ctrl+B (or Ctrl+W if supported by browser)
  - The filter on suggested file types can be disabled by pressing Ctrl+G.
  - If converted to an input, this functions as a string
- **seek_seconds**: An offset from the start of the sound file that the audio should start from
