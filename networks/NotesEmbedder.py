import torch
from torch import nn


class NotesEpisodeEmbedder(nn.Module):
    def __init__(self, in_dim: int=768, out_dim: int=128, projection_mid_layer: int=64) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, projection_mid_layer),
            nn.ReLU(),
            nn.Linear(projection_mid_layer, out_dim)
        )

        self.linear_layer = nn.Linear(out_dim * 3, out_dim)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        decollated_input = input.reshape((3,768))
        out = self.projection(decollated_input)
        out = out.reshape((1, self.out_dim* 3))
        return self.relu(self.linear_layer(out))

if __name__ == '__main__':
    pass