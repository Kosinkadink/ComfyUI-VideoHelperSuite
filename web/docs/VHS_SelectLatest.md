# Select Latest 🎥🅥🅗🅢

Experimental virtual node to select the most recently modified file from a given folder

Assists in the creation of workflows where outputs from one execution are used elsewhere in subsequent executions.

## Inputs

- **filename_prefix**: A path which can consist of a combination of folders and a prefix which candidate files must match
- **filename_postfix**: A string which chich the selected file must end with. Useful for limiting to a target extension.

## Outputs

- **Filename**: A string representing a file path to the most recently modified file.
