import numpy as np
import torch

#Relative positional encoding
def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_relative_pos_embed(embed_dim, grid_size)
    relative_pos = 2*np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos

# 2D sine-cosine postion embedding
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)  # [2, grid_size, grid_size]

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0, "embed_dim must be even"

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) #(H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) #(H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # [H*W, D]
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size(M,)
    out: (M,D)
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2 
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)  # [M, D/2], outer product

    emb_sin = np.sin(out)  # [M, D/2]
    emb_cos = np.cos(out)  # [M, D/2]

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [M, D]
    return emb