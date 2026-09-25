import torch
from torch import nn

from Networks.LabEmbedder import LabEpisodeEncoder
from Networks.MicrobiologyEmbedder import MicrobiologyEpisodeEncoder
from Networks.NotesEmbedder import NotesEpisodeEmbedder
from Networks.PrescriptionEmbedder import PrescriptionEpisodeEncoder
from Networks.RadiologyEmbedder import RadiologyEpisodeEmbedder


class EpisodeEmbedderModule(nn.Module):
    def __init__(self, device, num_prescriptions:int, num_labs: int, num_micro_labs: int, num_exam_codes: int, lab_embed_dim: int, output_dim: int):
        super().__init__()
        self.device = device
        self.note_encoder = NotesEpisodeEmbedder(out_dim=output_dim)
        self.lab_encoder = LabEpisodeEncoder(num_labs, lab_embed_dim, output_dim)
        self.microbiology_encoder = MicrobiologyEpisodeEncoder(num_micro_labs, lab_embed_dim, output_dim)
        self.prescription_encoder = PrescriptionEpisodeEncoder(num_prescriptions, embedding_dim=output_dim)
        self.radiology_encoder = RadiologyEpisodeEmbedder(num_exam_codes, lab_embed_dim, output_dim)

    def forward(self, interventions) -> torch.Tensor:
        embeddings = []

        for intervention in interventions:
            intervention_type = intervention['intervention_type']

            if intervention_type == 'notes':
                intervention_embedding = intervention['intervention_embedding'].to(device=self.device)
                embeddings.append(self.note_encoder(intervention_embedding))
            elif intervention_type == 'labevents':
                intervention_embedding = intervention['intervention_embedding']
                intervention_lab_ids_embedding = intervention_embedding[0].to(device=self.device)
                intervention_lab_values_embedding = intervention_embedding[1].to(device=self.device)
                embed = self.lab_encoder(intervention_lab_ids_embedding, intervention_lab_values_embedding)
                embeddings.append(embed)
            elif intervention_type == 'microbiologyevents':
                intervention_embedding = intervention['intervention_embedding']
                intervention_micro_lab_ids_embedding = intervention_embedding[0].to(device=self.device)
                intervention_micro_lab_values_embedding = intervention_embedding[1].to(device=self.device)
                embed = self.microbiology_encoder(intervention_micro_lab_ids_embedding, intervention_micro_lab_values_embedding)
                embeddings.append(embed)
            elif intervention_type == 'prescriptions':
                intervention_embedding = intervention['intervention_embedding'].to(device=self.device)
                embeddings.append(self.prescription_encoder(intervention_embedding))
            elif intervention_type == 'radiology':
                intervention_embedding = intervention['intervention_embedding']
                intervention_exam_code_embedding = intervention_embedding[0].to(device=self.device)
                intervention_values_embedding = intervention_embedding[1].to(device=self.device)
                embeddings.append(self.radiology_encoder(intervention_exam_code_embedding, intervention_values_embedding))

        return embeddings
if __name__ == '__main__':
    pass