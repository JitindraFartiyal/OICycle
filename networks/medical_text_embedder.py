import torch
import numpy as np

from torch import nn
from transformers import AutoTokenizer, AutoModel


class MedicalTextEmbedder(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def forward(self, texts: list[str]) -> np.ndarray:
        input_token = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        input_token = {
            key: value.to(self.device)
            for key, value in input_token.items()
        }

        output = self.model(**input_token)
        embedding = output.last_hidden_state[:, 0, :]

        return embedding.cpu().numpy()



if __name__ == "__main__":
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = ['Gender: M, Age: 24, Chief Complaint: Cold and Fever', 'Gender: F, Age: 35, Chief Complaint: Cold and Fever']
    med_text_embedder = MedicalTextEmbedder(torch_device)
    text_embedding = med_text_embedder(text)

    print(text_embedding.shape)