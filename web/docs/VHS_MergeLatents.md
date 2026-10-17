# Merge Latents 🎥🅥🅗🅢

Combine two groups of latents into a single group of latents

## Inputs

- **latents_A**: The first group of latents
- **latents_B**: The first group of latents

## Outputs

- **LATENT**: The combined group of latents
- **count**: The length of the combined group

## Widgets

- **merge_strategy**: Determines what the output resolution will be if input resolutions don't match
  - **match A**: Always use the resolution for A
  - **match B**: Always use the resolution for B
  - **match smaller**: Pick the smaller resolution by area
  - **match larger**: Pick the larger resolution by area
- **scale_method**: Determines what method to use if scaling is required
- **crop**: When sizes don't match, should the resized image have it's aspect ratio changed, or be cropped to maintain aspect ratio
