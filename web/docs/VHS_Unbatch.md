# Unbatch 🎥🅥🅗🅢

Unbatch a list of items into a single concatenated item

Useful for when you want a single video output from a complex workflow

Has no relation to the Meta Batch system of VHS

## Inputs

- **batched**: Any input which may or may not be batched

## Outputs

- **unbatched**: A single output element. Torch tensors are concatenated across dim 0, all other types are added which functions as concatenation for strings and arrays, but may give undesired results for other types
