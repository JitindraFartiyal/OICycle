import pickle
import time
import logging
import torch
import tqdm as tqdm
from Datasets.utils import get_preds_from_logits
from Metrics.evaluation_metrics import calculate_metrics
from TrainEval.config import config, model_type

logger = logging.getLogger(__name__)

def train_one_epoch(device, epoch, state_model, episode_embedding_model, init_state, loader, optimizer, loss_fn, acc_steps):
    state_model.train()
    episode_embedding_model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    # preds_probs = {}
    # targets = {}

    for batch_idx, batch in enumerate(loader):
        iter_start_time = time.time()
        subject_id = batch[0]['subject_id']
        episodical_data = batch[0]['episodical_data']
        gt = batch[0]['gt']

        total_episodes = len(episodical_data)
        previous_state = init_state
        patient_loss = torch.tensor(0.0).to(device)

        # preds_probs[subject_id] = {}
        # targets[subject_id] = {}
        prediction_ep_ct = 0
        for episode in range(total_episodes):
            prediction_state = episodical_data[episode]['prediction_state']
            prediction_type = episodical_data[episode]['prediction_type']
            interventions = episodical_data[episode]['interventions']

            episode_embeddings = episode_embedding_model(interventions)
            if model_type == 'transformers':
                episode_embeddings = torch.stack(episode_embeddings, dim=1)
                output = state_model(episode_embeddings)
            else:
                # for single clinical snapshot
                if model_type == 'single_snapshot':
                    previous_state = init_state

                for each_embedding in episode_embeddings:
                    output = state_model(each_embedding, previous_state, 'train')
                    previous_state = output['state']

            if prediction_state:
                prediction_ep_ct += 1

                time_to_next_episode_target = episodical_data[episode]['time_to_next_episode_target']
                mortality_target = episodical_data[episode]['mortality_target']
                target = gt[episode]

                # np_target = {}
                # for key, value in target.items():
                #     if isinstance(value, torch.Tensor):
                #         np_target[key] = value.detach().cpu().numpy()
                #     else:
                #         np_target[key] = value
                #
                # targets[subject_id][episode] = np_target

                curr_loss = loss_fn(prediction_type, output, target, time_to_next_episode_target, mortality_target)
                patient_loss = patient_loss + curr_loss

                # preds, probs = get_preds_from_logits(output, config['threshold'])
                # preds_probs[subject_id][episode] = probs

        if prediction_ep_ct > 0:
            patient_loss = patient_loss / prediction_ep_ct
            total_loss += patient_loss.item()
            (patient_loss / acc_steps).backward()

        if (batch_idx + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % config['print_every'] == 0:
            logger.info(f'Iters/Epoch: {batch_idx}/{epoch} | Loss: {round(total_loss / (batch_idx + 1), 4)} '
                  f'| iter time: {round((time.time()-iter_start_time) * config['print_every'], 4)}')

    # with open(fr"D:\data\Project Medical AI\results_v2.0\preds_probs_train_{model_type}.pkl", "wb") as file:
    #     pickle.dump(preds_probs, file)
    #
    # with open(fr"D:\data\Project Medical AI\results_v2.0\targets_train_{model_type}.pkl", "wb") as file:
    #     pickle.dump(targets, file)
    #
    # metrics = calculate_metrics(preds_probs, targets, config['threshold'])
    metrics = None
    return round(total_loss/len(loader), 4), metrics

def validate(state_model, episode_embedding_model, init_state, loader):
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

                if prediction_state:
                    target = gt[episode]
                    np_target = {}
                    for key, value in target.items():
                        if isinstance(value, torch.Tensor):
                            np_target[key] = value.detach().cpu().numpy()
                        else:
                            np_target[key] = value

                    targets[subject_id][episode] = np_target

                    preds, probs = get_preds_from_logits(output, config['threshold'])
                    preds_probs[subject_id][episode] = probs

        with open(fr"D:\data\Project Medical AI\results_v2.0\preds_probs_val_{model_type}.pkl", "wb") as file:
            pickle.dump(preds_probs, file)

        with open(fr"D:\data\Project Medical AI\results_v2.0\targets_val_{model_type}.pkl", "wb") as file:
            pickle.dump(targets, file)

        metrics = calculate_metrics(preds_probs, targets, config['threshold'])

        return metrics



if __name__ == '__main__':
    pass
