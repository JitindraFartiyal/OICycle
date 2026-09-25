import torch
import torch.nn as nn

class PrescriptionEmbedder(nn.Module):

    def __init__(self, num_prescription: int, embedding_dim: int=256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_prescription, embedding_dim=embedding_dim)

    def forward(self, drug_ids):
        return self.embedding(drug_ids)

class AttentionPooling(nn.Module):

    def __init__(self, hidden_dim: int=128):
        super().__init__()
        self.attention = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x):
        score = self.attention(x)
        weight = torch.softmax(score, dim=0)
        pooled = (weight * x).sum(dim=0)

        return pooled


class MeanPooling(nn.Module):

    def forward(self, x):
        return x.mean(dim=0)

class PrescriptionEpisodeEncoder(nn.Module):

    def __init__(self, num_prescriptions, embedding_dim) -> None:
        super().__init__()
        self.measurement_encoder = PrescriptionEmbedder(num_prescriptions, embedding_dim=embedding_dim)
        self.pool = AttentionPooling(hidden_dim=embedding_dim)
        # self.pool = MeanPooling()

    def forward(self, values):

        measurements = self.measurement_encoder(values)
        episode_embedding = self.pool(measurements)

        return episode_embedding