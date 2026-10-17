# Split Latents 🎥🅥🅗🅢

Split a set of latents into two groups

## Inputs

- **latents**: The latents to be split.

## Outputs

- **LATENT_A**: The first group of latents
- **A_count**: The number of latents in group A. This will be equal to split_index unless the latents input has length less than split_index
- **LATENT_B**: The second group of latents
- **B_count**: The number of latents in group B

## Widgets

- **split_index**: The index of the first latent that will be in the second output groups.
