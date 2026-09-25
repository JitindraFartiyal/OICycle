import os
import pickle

import numpy as np
import torch
import tqdm

from Datasets.observe_intervene_dataloader import get_dataloader
from Datasets.utils import get_preds_from_logits
from Metrics.evaluation_metrics import calculate_metrics
from Metrics.test_config import test_config, episodic_data_path, test_path, embedding_path, cxr_path, data_config, bsf, \
    model_type
from Networks.EpisodeEmbedder import EpisodeEmbedderModule
from Networks.StateModel import ClinicalStateMachineGRU
from Networks.StateTransformer import StateTransformer


def test_oci(state_model, episode_embedding_model, init_state, loader):
    state_model.eval()
    episode_embedding_model.eval()

    with torch.no_grad():
        preds_probs = {}
        targets = {}
        for batch_idx, batch in tqdm.tqdm(enumerate(loader)):
            subject_id = batch[0]['subject_id']
            episodical_data = batch[0]['episodical_data']
            gt = batch[0]['gt']

            total_episodes = len(episodical_data)
            previous_state = init_state

            preds_probs[subject_id] = {}
            targets[subject_id] = {}
            for episode in range(total_episodes):
                prediction_state = episodical_data[episode]['prediction_state']
                prediction_type = episodical_data[episode]['prediction_type']
                interventions = episodical_data[episode]['interventions']

                episode_embeddings = episode_embedding_model(interventions)
                if model_type == 'transformers':
                    embeddings = torch.stack(embeddings, dim=0)
                    embeddings = embeddings.unsqueeze(0)
                    output = state_model(embeddings)
                else:
                    # for single clinical snapshot
                    if model_type == 'single_snapshot':
                        previous_state = init_state

                    for each_embedding in episode_embeddings:
                        output = state_model(each_embedding, previous_state, 'test')
                        previous_state = output['state']

                np_output = output['state'].detach().cpu().numpy()
                belief_state_path = os.path.join(bsf, f'{subject_id}_ep_{episode}.npy')
                np.save(belief_state_path, np_output)

                if prediction_state:
                    target = gt[episode]
                    np_target = {}
                    for key, value in target.items():
                        if isinstance(value, torch.Tensor):
                            np_target[key] = value.detach().cpu().numpy()
                        else:
                            np_target[key] = value

                    targets[subject_id][episode] = np_target

                    preds, probs = get_preds_from_logits(output, test_config['threshold'])
                    preds_probs[subject_id][episode] = probs

        with open(fr"D:\data\Project Medical AI\results_v2.0\preds_probs_test_{model_type}.pkl", "wb") as file:
            pickle.dump(preds_probs, file)
        with open(fr"D:\data\Project Medical AI\results_v2.0\targets_test_{model_type}.pkl", "wb") as file:
            pickle.dump(targets, file)

        metrics = calculate_metrics(preds_probs, targets, test_config['threshold'])

        print('Task 1 metrics: \n', metrics['task1_metrics'])
        print('Task 2 metrics: \n', metrics['task2_metrics'])
        print('Task 3 metrics: \n', metrics['task3_metrics'])

        with open(fr"D:\data\Project Medical AI\results_v2.0\metrics_test_{model_type}.pkl", "wb") as file:
            pickle.dump(metrics, file)

        return metrics


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    test_dl = get_dataloader(episodic_data_path, test_path, embedding_path, cxr_path, size=-1)
    print(f'Test size: {len(test_dl)}')

    if model_type == 'transformers':
        state_model = StateTransformer(
            num_events=data_config['num_events'],
            num_subevents=data_config['num_subevents'],
            input_dim=test_config['hidden_dim'],
            d_model=test_config['hidden_dim']
        ).to(device)
    else:
        state_model = ClinicalStateMachineGRU(
            num_events=data_config['num_events'],
            num_subevents=data_config['num_subevents'],
            embed_dim=test_config['embed_dim'],
            hidden_dim=test_config['hidden_dim'],
        ).to(device)

    episode_embedding_model = EpisodeEmbedderModule(
        device=device,
        num_prescriptions=data_config['num_subevents']['n_pres'],
        num_labs=data_config['num_subevents']['n_lab'],
        num_micro_labs=data_config['num_subevents']['n_micro'],
        num_exam_codes=data_config['num_subevents']['n_radio'],
        lab_embed_dim=test_config['lab_embed_dim'],
        output_dim=test_config['embed_dim']
    ).to(device)

    state_model.load_state_dict(torch.load(test_config['load_state_model_path'], weights_only=True))
    episode_embedding_model.load_state_dict(torch.load(test_config['load_episode_embedding_model_path'], weights_only=True))
    init_state = torch.load(test_config['load_init_state_path'], weights_only=True)

    test_oci(state_model, episode_embedding_model, init_state, test_dl)

