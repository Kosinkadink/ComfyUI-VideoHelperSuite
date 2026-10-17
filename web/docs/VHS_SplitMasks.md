# Split Masks 🎥🅥🅗🅢

Split a set of masks into two groups

## Inputs

- **mask**: The masks to be split.

## Outputs

- **MASK_A**: The first group of masks
- **A_count**: The number of masks in group A. This will be equal to split_index unless the mask input has length less than split_index
- **MASK_B**: The second group of masks
- **B_count**: The number of masks in group B

## Widgets

- **split_index**: The index of the first latent that will be in the second output groups.
