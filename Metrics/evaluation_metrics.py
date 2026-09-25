import json
import numpy as np
from sklearn.metrics import f1_score


def get_task_1_metrics(y_prob: np.ndarray, y_true: np.ndarray):
    """
        AdaptiveRecall@k, Recall@10, Precision@10, ClinicalRelevance@10
    """
    adaptive_recall, recall_10, precision_10, clinical_relevance_10 = 0., 0., 0., 0.

    k = int(y_true.sum())

    # Adaptive Recall
    if k != 0:
        top_k = np.argsort(y_prob)[-k:]
        hits_k = y_true[top_k].sum()
        adaptive_recall = hits_k / k

        top_10 = np.argsort(y_prob)[-10:]
        hits_10 = y_true[top_10].sum()
        recall_10 = hits_10 / k

        precision_10 = hits_10 / 10

        clinical_relevance_10 = hits_10 / min(k, 10)

    return {
        'adaptive_recall': adaptive_recall,
        'recall_10': recall_10,
        'precision_10': precision_10,
        'clinical_relevance_10': clinical_relevance_10
    }


def get_task_1_auxiliary_metrics(y_prob: np.ndarray, y_true: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    return round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)


def get_task_2_metrics(y_prob: np.ndarray, y_true: np.ndarray):
    y_pred = np.argmax(y_prob, axis=-1)
    return round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)


def get_task_3_metrics(y_prob: np.ndarray, y_true: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    return round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)


def adaptive_recall_at_k(y_prob: np.ndarray, y_true: np.ndarray):
    k = int(y_true.sum())
    if k == 0:
        return 0.0
    else:
        top_k = np.argsort(y_prob)[-k:]
        hits = y_true[top_k].sum()
        return hits/k


def get_type_metrics(y_prob: np.ndarray, y_true: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    return round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)


def get_time_metrics(y_prob: np.ndarray, y_true: np.ndarray):
    y_pred = np.argmax(y_prob, axis=-1)
    return round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)


def get_mortality_metrics(y_prob: np.ndarray, y_true: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    return round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3)


def calculate_metrics(probs: dict, targets: dict, threshold: float=0.5, is_episode=False, ci=False):
    pres_K, lab_K, microbiology_K, radiology_K = [], [], [], []
    type_probs, type_targets = [], []
    time_probs, time_targets = [], []
    mortality_probs, mortality_targets = [], []
    task1_metrics = {
        'intervention_type_f1': 0.,
        'prescription': {'adaptive_recall': [], 'recall_10': [], 'precision_10': [], 'clinical_relevance_10': []},
        'lab': {'adaptive_recall': [], 'recall_10': [], 'precision_10': [], 'clinical_relevance_10': []},
        'microbiology': {'adaptive_recall': [], 'recall_10': [], 'precision_10': [], 'clinical_relevance_10': []},
        'radiology': {'adaptive_recall': [], 'recall_10': [], 'precision_10': [], 'clinical_relevance_10': []},
    }
    task2_metrics = {
        'intervention_time_f1': 0.
    }
    task3_metrics = {
        'mortality_time_f1': 0.
    }

    episode_level_metrics = {}

    for subject_id, prob in probs.items():
        max_episodes = len(prob)
        for episode_id, episode_probs in prob.items():
            ep_no = episode_id / max_episodes
            if ep_no not in episode_level_metrics.keys():
                episode_level_metrics[ep_no] = {
                    'task1': {'pres_recall': [], 'lab_recall': [], 'micro_recall': [], 'radio_recall': []},
                    'task2': {'time_probs': [], 'time_target': []},
                    'task3': {'mortality_probs': [], 'mortality_target': []},
                }

            target_val = targets[subject_id][episode_id]
            time_to_next_episode_target = target_val['time_to_next_episode_target']
            mortality_target = target_val['mortality_target']

            if time_to_next_episode_target != -1 and time_to_next_episode_target is not None:
                try:
                    time_targets.append(time_to_next_episode_target)
                    time_probs.append(episode_probs['time_to_next_episode_probs'])
                except Exception as e:
                    print(e)
                episode_level_metrics[ep_no]['task2']['time_probs'].append(episode_probs['time_to_next_episode_probs'])
                episode_level_metrics[ep_no]['task2']['time_target'].append(time_to_next_episode_target)

            if mortality_target != -1:
                mortality_targets.append(mortality_target)
                mortality_probs.append(episode_probs['mortality_probs'])

                episode_level_metrics[ep_no]['task3']['mortality_probs'].append(episode_probs['mortality_probs'])
                episode_level_metrics[ep_no]['task3']['mortality_target'].append(mortality_target)

            if target_val['type'] == 'TI':
                pres_probs = episode_probs['prescription_probs']
                pres_targets = target_val['prescription_target']
                pres_metrics = get_task_1_metrics(pres_probs, pres_targets)
                pres_recall = pres_metrics['adaptive_recall'].item()
                episode_level_metrics[ep_no]['task1']['pres_recall'].append(pres_recall)
                task1_metrics['prescription']['adaptive_recall'].append(pres_metrics['adaptive_recall'])
                task1_metrics['prescription']['recall_10'].append(pres_metrics['recall_10'])
                task1_metrics['prescription']['precision_10'].append(pres_metrics['precision_10'])
                task1_metrics['prescription']['clinical_relevance_10'].append(pres_metrics['clinical_relevance_10'])
                # pres_recall = adaptive_recall_at_k(pres_probs, pres_targets)
                pres_K.append(pres_recall)
            elif target_val['type'] == 'DI':
                type_target = target_val['intervention_types']
                type_prob = episode_probs['intervention_type_probs']
                type_targets.append(type_target)
                type_probs.append(type_prob)

                if type_target[0] == 1:
                    lab_probs = episode_probs['lab_probs']
                    lab_targets = target_val['lab_target']
                    lab_metrics = get_task_1_metrics(lab_probs, lab_targets)
                    lab_recall = lab_metrics['adaptive_recall'].item()
                    episode_level_metrics[ep_no]['task1']['lab_recall'].append(lab_recall)
                    task1_metrics['lab']['adaptive_recall'].append(lab_metrics['adaptive_recall'])
                    task1_metrics['lab']['recall_10'].append(lab_metrics['recall_10'])
                    task1_metrics['lab']['precision_10'].append(lab_metrics['precision_10'])
                    task1_metrics['lab']['clinical_relevance_10'].append(lab_metrics['clinical_relevance_10'])
                    lab_K.append(lab_recall)
                if type_target[1] == 1:
                    microbiology_probs = episode_probs['microbiology_probs']
                    microbiology_targets = target_val['microbiology_target']
                    microbiology_metrics = get_task_1_metrics(microbiology_probs, microbiology_targets)
                    microbiology_recall = microbiology_metrics['adaptive_recall'].item()
                    episode_level_metrics[ep_no]['task1']['micro_recall'].append(microbiology_recall)
                    task1_metrics['microbiology']['adaptive_recall'].append(microbiology_metrics['adaptive_recall'])
                    task1_metrics['microbiology']['recall_10'].append(microbiology_metrics['recall_10'])
                    task1_metrics['microbiology']['precision_10'].append(microbiology_metrics['precision_10'])
                    task1_metrics['microbiology']['clinical_relevance_10'].append(microbiology_metrics['clinical_relevance_10'])
                    microbiology_K.append(microbiology_recall)
                if type_target[2] == 1:
                    radiology_probs = episode_probs['radiology_probs']
                    radiology_targets = target_val['radiology_target']
                    radiology_metrics = get_task_1_metrics(radiology_probs, radiology_targets)
                    radiology_recall = radiology_metrics['adaptive_recall'].item()
                    episode_level_metrics[ep_no]['task1']['radio_recall'].append(radiology_recall)
                    task1_metrics['radiology']['adaptive_recall'].append(radiology_metrics['adaptive_recall'])
                    task1_metrics['radiology']['recall_10'].append(radiology_metrics['recall_10'])
                    task1_metrics['radiology']['precision_10'].append(radiology_metrics['precision_10'])
                    task1_metrics['radiology']['clinical_relevance_10'].append(
                        radiology_metrics['clinical_relevance_10'])
                    radiology_K.append(radiology_recall)


    mean_task1_metrics = {}
    for key, value in task1_metrics.items():
        if key == 'intervention_type_f1':
            mean_task1_metrics['intervention_type_f1'] = value
        else:
            mean_task1_metrics[key] = {}
            for _k, _v in value.items():
                mean_task1_metrics[key][_k] = np.mean(_v).item()


    mean_task1_metrics['intervention_type_f1'] = get_task_1_auxiliary_metrics(np.array(type_probs), np.array(type_targets), threshold)
    task2_metrics['intervention_time_f1'] = get_task_2_metrics(np.array(time_probs), np.array(time_targets))
    task3_metrics['mortality_time_f1'] = get_task_3_metrics(np.array(mortality_probs), np.array(mortality_targets), threshold)

    if ci:
        return task1_metrics, task2_metrics, task3_metrics

    if is_episode:
        return episode_level_metrics

    return {
        'task1_metrics': mean_task1_metrics,
        'task2_metrics': task2_metrics,
        'task3_metrics': task3_metrics,
    }


def save_predictions(name: str, preds: dict, targets: dict):
    pred_dict = {}

    for subject_id, episodical_preds in preds.items():
        pred_dict[subject_id] = {}
        for episode_id, episode_preds in episodical_preds.items():
            target_val = targets[subject_id][episode_id]
            pred_dict[subject_id][episode_id] = []
            if target_val['type'] == 'TI':
                pred_labels = np.argwhere(episode_preds['prescription_preds']==1).flatten().tolist()
                target_labels = np.argwhere(target_val['prescription_target']==1).flatten().tolist()
                pred_dict[subject_id][episode_id].append({'intervention_type': 'prescription',
                                                          'prediction': pred_labels,
                                                          'target': target_labels})
            elif target_val['type'] == 'DI':
                intervention_target = target_val['intervention_types']
                lab_target = target_val['lab_target']
                microbiology_target = target_val['microbiology_target']
                radiology_target = target_val['radiology_target']

                lab_label = np.argwhere(lab_target==1).flatten().tolist()
                microbiology_label = np.argwhere(microbiology_target==1).flatten().tolist()
                radiology_label = np.argwhere(radiology_target==1).flatten().tolist()


                if len(lab_label) > 0:
                    lab_pred_labels = np.argwhere(episode_preds['lab_preds']==1).flatten().tolist()
                    pred_dict[subject_id][episode_id].append({
                        'intervention_type': 'lab',
                        'prediction': lab_pred_labels,
                        'target': lab_label})

                if len(microbiology_label) > 0:
                    miro_pred_labels = np.argwhere(episode_preds['microbiology_preds']==1).flatten().tolist()
                    pred_dict[subject_id][episode_id].append({
                        'intervention_type': 'microbiology',
                        'prediction': miro_pred_labels,
                        'target': microbiology_label})

                if len(radiology_label) > 0:
                    radio_pred_labels = np.argwhere(episode_preds['radiology_preds']==1).flatten().tolist()
                    pred_dict[subject_id][episode_id].append({
                        'intervention_type': 'radiology',
                        'prediction': radio_pred_labels,
                        'target': radiology_label})

    with open(fr"D:\data\Project Medical AI\results_v2.0\predictions_{name}.json", "w") as f:
        json.dump(pred_dict, f, indent=2)


if __name__ == '__main__':
    pass