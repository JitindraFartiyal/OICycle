import numpy as np
import torch
from torch import nn

from TrainEval.config import config, intervention_type_weights_path, intervention_value_weights_path


class ObserveInterveneLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        intervention_type_pos_weight = np.load(intervention_type_weights_path, allow_pickle=True)['weights']
        intervention_value_pos_weight = np.load(intervention_value_weights_path, allow_pickle=True)['weights'].item()

        self.di_type_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(intervention_type_pos_weight))

        self.lab_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(intervention_value_pos_weight['labevents']))
        self.microbiology_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(intervention_value_pos_weight['microbiology']))
        self.radiology_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(intervention_value_pos_weight['radiology']))
        self.prescription_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(intervention_value_pos_weight['prescriptions']))

        self.time_to_next_episode_loss = nn.CrossEntropyLoss()
        self.mortality_loss = nn.BCEWithLogitsLoss()

        self.loss_weights = [1., 1., 1.]

    def forward(self, condition: str, logits: dict, targets: dict, time_to_next_episode_target, mortality_target):
        type_loss, pres_loss, lab_loss, micro_loss, radio_loss = 0., 0., 0., 0., 0.
        time_loss, mortality_loss = 0., 0.

        if time_to_next_episode_target != -1:
            time_to_next_episode_target = time_to_next_episode_target.to(self.device, dtype=torch.long)
            time_loss = self.time_to_next_episode_loss(logits['time_to_next_episode_prediction'], time_to_next_episode_target)

        mortality_target = mortality_target.to(self.device).unsqueeze(0)
        mortality_loss = self.mortality_loss(logits['mortality_prediction'], mortality_target)

        if condition == "TI":
            target = targets['prescription_target'].to(self.device).unsqueeze(0)
            pres_loss = self.prescription_loss(logits['prescription_preds'], target)
        elif condition == "DI":
            target = targets['intervention_types'].to(self.device).unsqueeze(0)
            type_loss  = self.di_type_loss(logits['intervention_type_prediction'], target)

            if targets['intervention_types'][0].item() == 1:
                target = targets['lab_target'].to(self.device).unsqueeze(0)
                lab_loss = self.lab_loss(logits['lab_preds'], target)

            if targets['intervention_types'][1].item() == 1:
                target = targets['microbiology_target'].to(self.device).unsqueeze(0)
                micro_loss = self.microbiology_loss(logits['microbiology_preds'], target)

            if targets['intervention_types'][2].item() == 1:
                target = targets['radiology_target'].to(self.device).unsqueeze(0)
                radio_loss = self.radiology_loss(logits['radiology_preds'], target)


        t1_loss = (
                config['lambdas']['loss_type'] * type_loss +
                config['lambdas']['loss_pres'] * pres_loss +
                config['lambdas']['loss_lab'] * lab_loss +
                config['lambdas']['loss_micro'] * micro_loss +
                config['lambdas']['loss_radio'] * radio_loss
        )
        t2_loss = time_loss
        t3_loss = mortality_loss

        combined_loss = (
            config['lambdas']['loss_t1'] * t1_loss +
            config['lambdas']['loss_t2'] * t2_loss +
            config['lambdas']['loss_t3'] * t3_loss
        )

        return combined_loss

if __name__ == '__main__':
    pass