import torch
import torch.nn as nn

class TileCoder(nn.Module):
    def __init__(self, tiles_per_dim, value_limits, tilings, offset_fn=None):
        super(TileCoder, self).__init__()
        if offset_fn is None:
            offset_fn = lambda n: 2 * torch.arange(n) + 1
        
        self._tiling_dims = torch.ceil(torch.tensor(tiles_per_dim, dtype=torch.float32)).to(torch.long) + 1
        self._offsets = self._calculate_offsets(len(tiles_per_dim), tilings, offset_fn)
        self._limits = torch.tensor(value_limits, dtype=torch.float32)
        self._norm_dims = torch.tensor(tiles_per_dim, dtype=torch.float32) / (self._limits[:, 1] - self._limits[:, 0])
        self._tile_base_ind = torch.prod(self._tiling_dims) * torch.arange(tilings).to(torch.long)
        self._hash_vec = self._calculate_hash_vec(self._tiling_dims)
        self._n_tiles = (tilings * torch.prod(self._tiling_dims)).item()
        self._tilings = tilings

    def _calculate_offsets(self, num_dims, tilings, offset_fn):
        base_offsets = torch.tensor(offset_fn(num_dims), dtype=torch.float32)
        offsets = torch.repeat_interleave(torch.arange(tilings).unsqueeze(0), num_dims, dim=0).T
        return (base_offsets * offsets / float(tilings)) % 1

    def _calculate_hash_vec(self, tiling_dims):
        return torch.tensor([torch.prod(tiling_dims[:i]).item() for i in range(len(tiling_dims))], dtype=torch.long)

    def forward(self, x):
        batch_size = x.size(0)
        off_coords = ((x.unsqueeze(1) - self._limits[:, 0]) * self._norm_dims + self._offsets).to(torch.long)
        tile_indices = self._tile_base_ind.unsqueeze(0) + torch.matmul(off_coords, self._hash_vec)
        return tile_indices.view(batch_size, -1)

    @property
    def n_tiles(self):
        return self._n_tiles

    def inverse(self, tile_indices):
        coords = []
        batch_size = tile_indices.size(0)
        for ind in tile_indices.view(-1):
            tiling_index = ind // torch.prod(self._tiling_dims)
            ind = ind - self._tile_base_ind[tiling_index]
            coords.append(self._calculate_coord(ind))
        
        coords = torch.stack(coords).float()
        expanded_offsets = self._offsets.unsqueeze(0).expand(batch_size, *self._offsets.size())
        expanded_offsets = expanded_offsets.reshape(-1, expanded_offsets.size(-1))
        centers = (coords + 0.5 - expanded_offsets) / self._norm_dims + self._limits[:, 0]
        return torch.mean(centers.view(batch_size, self._tilings, -1), dim=1)

    def _calculate_coord(self, ind):
        coord = torch.zeros(len(self._tiling_dims), dtype=torch.long)
        for dim in reversed(range(len(self._tiling_dims))):
            divisor = torch.prod(self._tiling_dims[:dim]).item()
            coord[dim] = ind // divisor
            ind = ind % divisor
        return coord
