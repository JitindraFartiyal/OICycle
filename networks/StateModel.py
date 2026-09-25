import torch
import torch.nn as nn

from TrainEval.config import model_type


class PatientStateModelGRU(nn.Module):
    def __init__(self, input_dim: int=128, hidden_dim: int=768) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim)

    def forward(self, episode_embedding: torch.Tensor, previous_state: torch.Tensor) -> torch.Tensor:
        new_state = self.gru(episode_embedding, previous_state)

        return new_state


class InterventionTypePredictor(nn.Module):
    def __init__(self, hidden_dim: int=256, out_feats: int=64, num_events: int=3) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=out_feats),
            nn.ReLU(),
            nn.Linear(in_features=out_feats, out_features=num_events)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.predictor(state)


class BeliefNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.pdf = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.mu = nn.Linear(hidden_dim, hidden_dim)
        self.logvar = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, mode:str='train'):
        h = self.pdf(h)
        mu = self.mu(h)
        logvar = self.logvar(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        if mode == 'train':
            z = mu + std * eps
        else:
            z = mu

        return z, mu, logvar


class LabEventsPredictor(nn.Module):

    def __init__(self, num_subevents: int, hidden_dim: int=256, out_feats: int=64) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=out_feats),
            nn.ReLU(),
            nn.Linear(in_features=out_feats, out_features=num_subevents)
        )

    def forward(self, state):
        return self.predictor(state)


class MicrobiologyPredictor(nn.Module):

    def __init__(self, num_subevents: int, hidden_dim: int=256, out_feats: int=64) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=out_feats),
            nn.ReLU(),
            nn.Linear(in_features=out_feats, out_features=num_subevents)
        )

    def forward(self, state):
        return self.predictor(state)


class PrescriptionPredictor(nn.Module):

    def __init__(self, num_subevents: int, hidden_dim: int=256, out_feats: int=64) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=out_feats),
            nn.ReLU(),
            nn.Linear(in_features=out_feats, out_features=num_subevents)
        )

    def forward(self, state):
        return self.predictor(state)


class RadiologyPredictor(nn.Module):

    def __init__(self, num_subevents: int, hidden_dim: int=256, out_feats: int=64) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=out_feats),
            nn.ReLU(),
            nn.Linear(in_features=out_feats, out_features=num_subevents)
        )

    def forward(self, state):
        return self.predictor(state)


class TimeToNextEpisodePredictor(nn.Module):
    def __init__(self, hidden_dim: int, out_feats: int=5) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=out_feats)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)


class MortalityIn24hrsPredictor(nn.Module):
    def __init__(self, hidden_dim: int, out_feats: int=1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=out_feats)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)


class ClinicalStateMachineGRU(nn.Module):
    def __init__(self, num_events: int, num_subevents: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_model = PatientStateModelGRU(input_dim=embed_dim, hidden_dim=hidden_dim)
        self.belief_model = BeliefNetwork(hidden_dim=hidden_dim)
        self.intervention_type_predictor = InterventionTypePredictor(hidden_dim=hidden_dim, num_events=num_events)

        self.labevents_predictor = LabEventsPredictor(hidden_dim=hidden_dim, num_subevents=num_subevents['n_lab'])
        self.microbiology_predictor = MicrobiologyPredictor(hidden_dim=hidden_dim, num_subevents=num_subevents['n_micro'])
        self.prescription_predictor = PrescriptionPredictor(hidden_dim=hidden_dim, num_subevents=num_subevents['n_pres'])
        self.radiology_predictor = RadiologyPredictor(hidden_dim=hidden_dim, num_subevents=num_subevents['n_radio'])

        self.time_to_next_episode_predictor = TimeToNextEpisodePredictor(hidden_dim=hidden_dim)
        self.mortality_24hrs_predictor = MortalityIn24hrsPredictor(hidden_dim=hidden_dim)

    def forward(self, episode_embedding: torch.Tensor, previous_state: torch.Tensor, mode:str='train') -> torch.Tensor:

        state = self.state_model(episode_embedding, previous_state)
        if model_type == 'latent':
            belief_state = state
        else:
            belief_state, _, _ = self.belief_model(state, mode)

        intervention_type_prediction = self.intervention_type_predictor(belief_state)

        lab_preds = self.labevents_predictor(belief_state)
        micro_preds = self.microbiology_predictor(belief_state)
        pres_preds = self.prescription_predictor(belief_state)
        radio_preds = self.radiology_predictor(belief_state)
        time_to_next_episode_prediction = self.time_to_next_episode_predictor(belief_state)
        mortality_prediction = self.mortality_24hrs_predictor(belief_state)

        return {
            'state': state,
            'intervention_type_prediction': intervention_type_prediction,
            'lab_preds': lab_preds,
            'microbiology_preds': micro_preds,
            'prescription_preds': pres_preds,
            'radiology_preds': radio_preds,
            'time_to_next_episode_prediction': time_to_next_episode_prediction,
            'mortality_prediction': mortality_prediction,
        }