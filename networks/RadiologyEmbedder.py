import torch
from torch import nn


class RadiologyEmbedder(nn.Module):
    def __init__(self, nums_exam_code: int, exam_code_embed_dim: int=64, output_dim: int=256) -> None:
        super().__init__()
        self.exam_code_embedding = nn.Embedding(num_embeddings=nums_exam_code, embedding_dim=exam_code_embed_dim, padding_idx=0)
        self.pool = AttentionPooling(hidden_dim=exam_code_embed_dim)

        self.embedder = nn.Sequential(
            nn.Linear(exam_code_embed_dim + 768, exam_code_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(exam_code_embed_dim * 2, output_dim)
        )

    def forward(self, exam_codes, values) -> torch.Tensor:

        exam_code_emb = self.exam_code_embedding(exam_codes)
        exam_code_emb = self.pool(exam_code_emb)
        exam_code_emb = exam_code_emb.squeeze(0)

        x = torch.cat([exam_code_emb, values], dim=1)

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


class RadiologyEpisodeEmbedder(nn.Module):

    def __init__(self, nums_exam_code: int, exam_code_embed_dim: int, output_dim: int):
        super().__init__()
        self.measurement_encoder = RadiologyEmbedder(nums_exam_code, exam_code_embed_dim=exam_code_embed_dim, output_dim=output_dim)


    def forward(self, exam_codes, values):
        episode_embedding = self.measurement_encoder(exam_codes, values)

        return episode_embedding


if __name__ == '__main__':
    pass