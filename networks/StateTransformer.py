import torch.nn as nn

from Networks.StateModel import InterventionTypePredictor, LabEventsPredictor, MicrobiologyPredictor, \
    PrescriptionPredictor, RadiologyPredictor, TimeToNextEpisodePredictor, MortalityIn24hrsPredictor


class StateTransformer(nn.Module):
    def __init__(self, num_events, num_subevents, input_dim=512, hidden_dim=512, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.intervention_type_predictor = InterventionTypePredictor(hidden_dim=hidden_dim, num_events=num_events)

        self.labevents_predictor = LabEventsPredictor(hidden_dim=hidden_dim, num_subevents=num_subevents['n_lab'])
        self.microbiology_predictor = MicrobiologyPredictor(hidden_dim=hidden_dim, num_subevents=num_subevents['n_micro'])
        self.prescription_predictor = PrescriptionPredictor(hidden_dim=hidden_dim, num_subevents=num_subevents['n_pres'])
        self.radiology_predictor = RadiologyPredictor(hidden_dim=hidden_dim, num_subevents=num_subevents['n_radio'])

        self.time_to_next_episode_predictor = TimeToNextEpisodePredictor(hidden_dim=hidden_dim)
        self.mortality_24hrs_predictor = MortalityIn24hrsPredictor(hidden_dim=hidden_dim)


    def forward(self, embeddings, padding_mask=None):
        """
        embeddings: [batch, num_embeddings, input_dim]
        padding_mask: [batch, num_embeddings] True = padding, False = valid token
        """
        x = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        intervention_type_prediction = self.intervention_type_predictor(x)
        lab_preds = self.labevents_predictor(x)
        micro_preds = self.microbiology_predictor(x)
        pres_preds = self.prescription_predictor(x)
        radio_preds = self.radiology_predictor(x)
        time_to_next_episode_prediction = self.time_to_next_episode_predictor(x)
        mortality_prediction = self.mortality_24hrs_predictor(x)

        return {
            'state': x,
            'intervention_type_prediction': intervention_type_prediction,
            'lab_preds': lab_preds,
            'microbiology_preds': micro_preds,
            'prescription_preds': pres_preds,
            'radiology_preds': radio_preds,
            'time_to_next_episode_prediction': time_to_next_episode_prediction,
            'mortality_prediction': mortality_prediction,
        }