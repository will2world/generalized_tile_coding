# User Guide: Using the Generalized tile coding module in python

## Overview
The TileCoder class encodes multi-dimensional continuous input data into discrete tile indices, which can be used for reinforcement learning applications. It also supports decoding these indices back into approximate original values. This guide will show you how to initialize, use, and understand the key methods in the TileCoder class.

## Initialization
To initialize the TileCoder class, you need to specify the number of tiles per dimension, the value limits for each dimension, and the number of tilings.
```python
tiles_per_dim = [10, 10, 10, 10]  # Number of tiles per dimension
value_limits = [[0, 1], [0, 1], [0, 1], [0, 1]]  # Value limits for each dimension
tilings = 4  # Number of tilings

tile_coder = TileCoder(tiles_per_dim, value_limits, tilings)
```

## Methods
### 1. Forward Method
The forward method encodes input data into tile indices.
```python
inputs = torch.tensor([[0.5, 0.5, 0.5, 0.1], [0.1, 0.9, 0.2, 0.1], [0.45, 0.354, 0.77, 0.1]], dtype=torch.float32)
encoded_inputs = tile_coder(inputs)
print("Encoded Inputs:")
print(encoded_inputs)
```

### 2. Inverse Method
The inverse method decodes the tile indices back into approximate original values.
```python
decoded_inputs = tile_coder.inverse(encoded_inputs)
print("Decoded Inputs:")
print(decoded_inputs)
```

## Example
The following example shows how to initialize the TileCoder, encode inputs, decode the tile indices, and compute the cosine similarity between the original and decoded inputs.
```python
import torch.nn.functional as F

# Initialize TileCoder
tiles_per_dim = [10, 10, 10, 10]
value_limits = [[0, 1], [0, 1], [0, 1], [0, 1]]
tilings = 4
tile_coder = TileCoder(tiles_per_dim, value_limits, tilings)

# Example input
inputs = torch.tensor([[0.5, 0.5, 0.5, 0.1], [0.1, 0.9, 0.2, 0.1], [0.45, 0.354, 0.77, 0.1]], dtype=torch.float32)

# Encode inputs
encoded_inputs = tile_coder(inputs)
print("Encoded Inputs:")
print(encoded_inputs)

# Decode inputs
decoded_inputs = tile_coder.inverse(encoded_inputs)
print("Decoded Inputs:")
print(decoded_inputs)

# Compute cosine similarity
cosine_similarity = F.cosine_similarity(inputs, decoded_inputs)
print("Cosine Similarity:")
print(cosine_similarity)
```

## Explanation of Methods
- \_\_init\_\_: Initializes the tile coder with the specified number of tiles per dimension, value limits, and number of tilings.
- \_calculate_offsets: Computes the offsets for the tilings.
- \_calculate_hash_vec: Computes the hash vector for indexing.
- forward: Encodes the input data into tile indices.
- inverse: Decodes the tile indices back into approximate original values.
- \_calculate_coord: Computes the coordinates from a given index.
