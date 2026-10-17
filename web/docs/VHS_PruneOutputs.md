# Prune Outputs 🎥🅥🅗🅢

Automates deletion of undesired outputs from a Video Combine node.

Video Combine produces a number of file outputs in addition to the final output. Some of these, such as a video file without audio included, are implementation limitations and are not feasible to solve. As an alternative, the Prune Outputs node is added to automate the deletion of these file outputs if they are not desired

## Inputs

- **filenames**: A connection from a Video Combine node to indicate which outputs should be pruned

## Widgets

- **options**: Which files should be deleted
  - **Intermediate**: Delete any files that were required for intermediate processing but are not the final output, like the no-audio output file when audio is included
  - **Intermediate and Utility**: Delete all produced files that aren't the final output, including the first frame png
