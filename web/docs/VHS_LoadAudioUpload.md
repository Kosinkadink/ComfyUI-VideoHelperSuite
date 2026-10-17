# Load Audio (Upload) 🎥🅥🅗🅢

Loads an audio file from the input directory

Very similar in functionality to the built-in LoadAudio. It was originally added before VHS swapped to use Comfy's internal AUDIO format, but provides the additional options for start time and duration

## Outputs

- **audio**: The loaded audio

## Widgets

- **audio**: The audio file to be loaded.
- **start_time**: An offset from the start of the sound file that the audio should start from
- **duration**: A maximum limit for the audio. Disabled if 0
- **choose audio to upload**: An upload button is provided to upload an audio file to the input folder
