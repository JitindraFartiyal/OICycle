import torch
import torch.nn as nn


class LabEmbedder(nn.Module):

    def __init__(self, num_labs: int, lab_embed_dim: int=64, output_dim: int=256) -> None:
        super().__init__()

        self.lab_embedding = nn.Embedding(num_embeddings=num_labs, embedding_dim=lab_embed_dim)

        self.embedder = nn.Sequential(
            nn.Linear(in_features=lab_embed_dim + 1, out_features=lab_embed_dim*2),
            nn.ReLU(),
            nn.Linear(in_features=lab_embed_dim*2, out_features=output_dim)
        )

    def forward(self, lab_id, value):
        lab_emb = self.lab_embedding(lab_id)
        lab_emb = lab_emb.squeeze(1)
        x = torch.cat([lab_emb, value],dim=1)

        embedding = self.embedder(x)

        return embedding

class AttentionPooling(nn.Module):

    def __init__(self, hidden_dim: int=128):
        super().__init__()
        self.attention = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x):
        score = self.attention(x)
        weight = torch.softmax(score, dim=0)
        pooled = (weight * x).sum(dim=0)

        return pooled.unsqueeze(0)

class LabEpisodeEncoder(nn.Module):

    def __init__(self, num_labs: int, lab_embed_dim: int, output_dim: int):
        super().__init__()
        self.measurement_encoder = LabEmbedder(num_labs, lab_embed_dim=lab_embed_dim, output_dim=output_dim)
        self.pool = AttentionPooling(hidden_dim=output_dim)

    def forward(self, lab_ids, values):
        measurements = self.measurement_encoder(lab_ids, values)
        episode_embedding = self.pool(measurements)

        return episode_embedding