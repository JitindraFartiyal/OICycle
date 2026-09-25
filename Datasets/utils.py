import re
from TrainEval.config import data_config, labevents_stats, microbiology_stats, prescriptions_stats, radiology_stats
import pandas as pd
import torch


def labevents_preprocessor(labevents: pd.Series) -> (list, list):
    labevents_df = pd.DataFrame(labevents, columns=['intervention_value'])
    labevents_df[['itemid', 'values']] = labevents_df['intervention_value'] .str.split(',', n=1, expand=True)

    labevents_df = labevents_df.merge(
        labevents_stats,
        on='itemid',
        how='left',
    )[['itemid', 'values', 'index', 'ref_range_lower', 'ref_range_upper']]
    labevents_df = labevents_df.astype({'itemid':'int64', 'values':'float64', 'ref_range_lower':'float64', 'ref_range_upper':'float64', 'index':'int64'})

    normalized_ref = (labevents_df['ref_range_lower'] + labevents_df['ref_range_upper']) / 2
    denom = (labevents_df['ref_range_upper'] - labevents_df['ref_range_lower']) + 1e-9
    denom[denom == 0.] = 1.
    labevents_df['normalized_lab_value'] = (labevents_df['values'] - normalized_ref) / denom

    return labevents_df['index'].tolist(), labevents_df['normalized_lab_value'].tolist()

def microbiologyevents_preprocessor(microbiologyevents: pd.Series):
    microbiologyevents_df = pd.DataFrame(microbiologyevents, columns=['intervention_value'])
    microbiologyevents_df[['itemid', 'values']] = microbiologyevents_df['intervention_value'].str.split(',', n=1, expand=True)

    microbiologyevents_df = microbiologyevents_df.merge(
        microbiology_stats,
        on='itemid',
        how='left',
    )
    return microbiologyevents_df['index'].tolist(), microbiologyevents_df['value'].tolist()


def prescriptions_preprocessor(prescriptions: pd.Series):
    prescriptions_df = pd.DataFrame(prescriptions).rename(columns={'intervention_value':'itemid'})
    prescriptions_df = prescriptions_df.merge(
        prescriptions_stats,
        on='itemid',
        how='left',
    )
    return prescriptions_df['index'].tolist()


def radiology_preprocessor(radiology: pd.Series):
    values = radiology.str.split(',').values[0]

    embed_id = int(values[0])
    exam_codes = [radiology_stats[radiology_stats['itemid'] == code]['index'].item() for code in values[1:] if bool(re.search(r'[a-zA-Z]', code))]
    dicom_embed_idx = [int(_idx) for _idx in values[1:] if not bool(re.search(r'[a-zA-Z]', _idx))]

    return embed_id, exam_codes, dicom_embed_idx


def get_prescription_ground_truth(prescriptions: torch.Tensor):
    prescriptions_gt = torch.zeros(len(prescriptions_stats))
    prescriptions_gt[prescriptions] = 1.

    return prescriptions_gt


def get_labevents_ground_truth(labevents: torch.Tensor):
    labevents_itemid = labevents[0]
    labevents_gt = torch.zeros(len(labevents_stats))
    labevents_gt[labevents_itemid] = 1.

    return labevents_gt


def get_microbiologyevents_ground_truth(microbiologyevents: torch.Tensor):
    microbiologyevents_itemid = microbiologyevents[0]
    microbiologyevents_gt = torch.zeros(len(microbiology_stats))
    microbiologyevents_gt[microbiologyevents_itemid] = 1.

    return microbiologyevents_gt


def get_radiology_ground_truth(radiology: torch.Tensor):
    radiology_itemid = radiology[0]
    radiology_gt = torch.zeros(len(radiology_stats))
    radiology_gt[radiology_itemid] = 1.

    return radiology_gt


def get_ground_truth(episodical_data: list):
    episodical_gt = {}
    for i in range(len(episodical_data)):
        gt = {
            'type': None,
            'intervention_types': torch.Tensor([0., 0., 0.]),  # labevents, microbiologyevents, radiology
            'lab_target': torch.zeros(data_config['num_subevents']['n_lab']),
            'microbiology_target': torch.zeros(data_config['num_subevents']['n_micro']),
            'radiology_target': torch.zeros(data_config['num_subevents']['n_radio']),
            'prescription_target': torch.zeros(data_config['num_subevents']['n_pres']),
            'time_to_next_episode_target': episodical_data[i]['time_to_next_episode_target'],
            'mortality_target': episodical_data[i]['mortality_target']
        }
        if episodical_data[i]['prediction_state']:
            if episodical_data[i]['prediction_type'] == 'TI':
                gt['type'] = 'TI'
                gt['intervention_types'] = None
                for episode_intervention_value in episodical_data[i+1]['interventions']:
                    if episode_intervention_value['intervention_type'] == 'prescriptions':
                        gt['prescription_target'] = get_prescription_ground_truth(episode_intervention_value['intervention_embedding'])
                    else:
                        print('Should not go here as there is only prescriptions.')
            else:
                gt['type'] = 'DI'
                for episode_intervention_value in episodical_data[i + 1]['interventions']:
                    if episode_intervention_value['intervention_type'] == 'labevents':
                        gt['intervention_types'][0] = 1
                        gt['lab_target'] = get_labevents_ground_truth(episode_intervention_value['intervention_embedding'])
                    elif episode_intervention_value['intervention_type'] == 'microbiologyevents':
                        gt['intervention_types'][1] = 1
                        gt['microbiology_target'] = get_microbiologyevents_ground_truth(episode_intervention_value['intervention_embedding'])
                    elif episode_intervention_value['intervention_type'] == 'radiology':
                        gt['intervention_types'][2] = 1
                        gt['radiology_target'] = get_radiology_ground_truth(episode_intervention_value['intervention_embedding'])
                    else:
                        print('Error it should be between labevents, microbiologyevents and radiology.')
            episodical_gt[i] = gt

    return episodical_gt


def logits_to_preds(logits: torch.Tensor, threshold: float=0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = torch.nan_to_num(probs, nan=0.0)
    preds = (probs > threshold).float()

    return (
        probs.squeeze(0).detach().cpu().numpy(),
        preds.squeeze(0).detach().cpu().numpy()
    )


def get_preds_from_logits(logits: torch.Tensor, threshold: float=0.5) -> torch.Tensor:
    intervention_type_logits = logits['intervention_type_prediction'].detach().cpu()
    intervention_type_probs, intervention_type_preds = logits_to_preds(intervention_type_logits, threshold)

    time_to_next_episode_probs = logits['time_to_next_episode_prediction'].detach().cpu()
    time_to_next_episode_preds = time_to_next_episode_probs.argmax(dim=-1)

    mortality_logits = logits['mortality_prediction'].detach().cpu()
    mortality_probs, mortality_preds = logits_to_preds(mortality_logits, threshold)

    lab_logits = logits['lab_preds'].detach().cpu()
    lab_probs, lab_preds = logits_to_preds(lab_logits, threshold)

    microbiology_logits = logits['microbiology_preds'].detach().cpu()
    microbiology_probs, microbiology_preds = logits_to_preds(microbiology_logits, threshold)

    prescription_logits = logits['prescription_preds'].detach().cpu()
    prescription_probs, prescription_preds = logits_to_preds(prescription_logits, threshold)

    radiology_logits = logits['radiology_preds'].detach().cpu()
    radiology_probs, radiology_preds = logits_to_preds(radiology_logits, threshold)

    return ({
        'intervention_type_prediction': intervention_type_preds.numpy() if type(intervention_type_preds) is torch.Tensor else intervention_type_preds,
        'lab_preds': lab_preds.numpy() if type(lab_preds) is torch.Tensor else lab_preds,
        'microbiology_preds': microbiology_preds.numpy() if type(microbiology_preds) is torch.Tensor else microbiology_preds,
        'prescription_preds': prescription_preds.numpy() if type(prescription_preds) is torch.Tensor else prescription_preds,
        'radiology_preds': radiology_preds.numpy() if type(radiology_preds) is torch.Tensor else radiology_preds,
        'time_to_next_episode_prediction': time_to_next_episode_preds.numpy() if type(time_to_next_episode_preds) is torch.Tensor else time_to_next_episode_preds,
        'mortality_preds': mortality_preds.numpy() if type(mortality_preds) is torch.Tensor else mortality_preds,
    },
    {
        'intervention_type_probs': intervention_type_probs.numpy() if type(intervention_type_probs) is torch.Tensor else intervention_type_probs,
        'lab_probs': lab_probs.numpy() if type(lab_probs) is torch.Tensor else lab_probs,
        'microbiology_probs': microbiology_probs.numpy() if type(microbiology_probs) is torch.Tensor else microbiology_probs,
        'prescription_probs': prescription_probs.numpy() if type(prescription_probs) is torch.Tensor else prescription_probs,
        'radiology_probs': radiology_probs.numpy() if type(radiology_probs) is torch.Tensor else radiology_probs,
        'time_to_next_episode_probs': time_to_next_episode_probs.numpy() if type(time_to_next_episode_probs) is torch.Tensor else time_to_next_episode_probs,
        'mortality_probs': mortality_probs.numpy() if type(mortality_probs) is torch.Tensor else mortality_probs
    })


if __name__ == '__main__':
    pass