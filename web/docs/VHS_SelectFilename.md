# VAE Select Filename 🎥🅥🅗🅢

Select a single filename from the VHS_FILENAMES output by a Video Combine and return it as a string

Take care when combining this node with Prune Outputs. The VHS_FILENAMES object is immutable and will always contain the full list of output files, but execution order is undefined behavior (currently, Prune Outputs will generally execute first) and SelectFilename may return a path to a file that no longer exists.

## Inputs

- **filenames**: A VHS_FILENAMES from a Video Combine node

## Outputs

- **filename**: A string representation of the full output path for the chosen file

## Widgets

- **index**: The index of which file should be selected. The default, -1, chooses the most complete output
