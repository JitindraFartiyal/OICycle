import numpy as np
import pandas as pd
from TrainEval.config import labevents_stats, microbiology_stats, prescriptions_stats, radiology_stats, \
    intervention_type_weights_path, intervention_value_weights_path


def get_labevents_weights(df: pd.DataFrame, max_clip):
    lab_df = df[df['intervention_type'] == 'labevents'][
        ['subject_id', 'medical_episode', 'intervention_value']
    ].copy()

    total_lab_episodes = df[df['intervention_type'] == 'labevents'].groupby('subject_id')['medical_episode'].nunique().sum().item()

    lab_df['lab_items'] = lab_df['intervention_value'].apply(
        lambda x: set(x.split(',')[0::2])
    )

    episode_labs = (
        lab_df
        .groupby(['subject_id', 'medical_episode'])['lab_items']
        .agg(lambda x: set().union(*x))
    )

    labevents_subtypes = (
        episode_labs
        .explode()
        .value_counts()
        .to_dict()
    )

    item_to_idx = labevents_stats[['itemid', 'index']]
    labevents_positive_weights = convert_into_positive_weights(item_to_idx, labevents_subtypes, total_lab_episodes, max_clip)

    return labevents_positive_weights


def get_prescriptions_weights(df: pd.DataFrame, max_clip):
    prescriptions_df = df[df['intervention_type'] == 'prescriptions'][
        ['subject_id', 'medical_episode', 'intervention_value']
    ].copy()

    total_prescriptions_episodes = df[df['intervention_type'] == 'prescriptions'].groupby('subject_id')['medical_episode'].nunique().sum().item()

    prescriptions_df['items'] = prescriptions_df['intervention_value'].apply(
        lambda x: {x}
    )

    episode_prescriptions = (
        prescriptions_df
        .groupby(['subject_id', 'medical_episode'])['items']
        .agg(lambda x: set().union(*x))
    )
    prescriptions_subtypes = (
        episode_prescriptions
        .explode()
        .value_counts()
        .to_dict()
    )

    item_to_idx = prescriptions_stats[['itemid', 'index']]
    prescriptions_positive_weights = convert_into_positive_weights(item_to_idx, prescriptions_subtypes, total_prescriptions_episodes, max_clip)

    return prescriptions_positive_weights


def get_microbiology_weights(df: pd.DataFrame, max_clip):
    microbiology_df = df[df['intervention_type'] == 'microbiologyevents'][
        ['subject_id', 'medical_episode', 'intervention_value']
    ].copy()

    total_microbiology_episodes = df[df['intervention_type'] == 'microbiologyevents'].groupby('subject_id')['medical_episode'].nunique().sum().item()

    microbiology_df['items'] = microbiology_df['intervention_value'].apply(
        lambda x: set(x.split(',')[0::2])
    )

    episode_microbiology = (
        microbiology_df
        .groupby(['subject_id', 'medical_episode'])['items']
        .agg(lambda x: set().union(*x))
    )
    microbiology_subtypes = (
        episode_microbiology
        .explode()
        .value_counts()
        .to_dict()
    )

    item_to_idx = microbiology_stats[['itemid', 'index']]
    microbiology_positive_weights = convert_into_positive_weights(item_to_idx, microbiology_subtypes, total_microbiology_episodes, max_clip)

    return microbiology_positive_weights


def get_radiology_weights(df: pd.DataFrame, max_clip):
    radiology_df = df[df['intervention_type'] == 'radiology'][
        ['subject_id', 'medical_episode', 'intervention_value']
    ].copy()

    total_radiology_episodes = df[df['intervention_type'] == 'radiology'].groupby('subject_id')['medical_episode'].nunique().sum().item()

    radiology_df['items'] = radiology_df['intervention_value'].apply(
        lambda x: set(t for t in x.split(',')[1:] if len(t) < 10)
    )

    episode_radiology = (
        radiology_df
        .groupby(['subject_id', 'medical_episode'])['items']
        .agg(lambda x: set().union(*x))
    )
    radiology_subtypes = (
        episode_radiology
        .explode()
        .value_counts()
        .to_dict()
    )

    item_to_idx = radiology_stats[['itemid', 'index']]
    radiology_positive_weights = convert_into_positive_weights(item_to_idx, radiology_subtypes, total_radiology_episodes, max_clip)

    return radiology_positive_weights


def convert_into_positive_weights(itemid_to_idx: dict, event_subtypes: dict, total_event_episode: int, max_clip: float = 50.0):
    pos_weights = np.zeros(len(itemid_to_idx), dtype=np.float32)

    for i, row in itemid_to_idx.iterrows():
        itemid = row['itemid']
        idx = row['index']
        positive_count = event_subtypes.get(itemid, 0)

        if positive_count > 0:
            pos_weights[idx] = (total_event_episode - positive_count) / positive_count
        else:
            pos_weights[idx] = 0.0

    pos_weights = np.clip(pos_weights, 1.0, max_clip)

    return pos_weights


def get_intervention_type_weights(df: pd.DataFrame, total_episode:int, max_clip: float = 50.0):
    labevents_ct, microbiology_ct, radiology_ct = 0, 0, 0

    for i, row in df.iterrows():
        unique_intervention = set(row['intervention_type'])
        if 'labevents' in unique_intervention:
            labevents_ct += 1
        elif 'microbiologyevents' in unique_intervention:
            microbiology_ct += 1
        elif 'radiology' in unique_intervention:
            radiology_ct += 1

    intervention_type_pos_weight = np.array(
        [(total_episode - ct) / ct for ct in [labevents_ct, microbiology_ct, radiology_ct]])
    intervention_type_pos_weight = np.clip(intervention_type_pos_weight, 1.0, max_clip)

    return intervention_type_pos_weight


def get_intervention_value_weights(df: pd.DataFrame, max_clip):
    labevents_weights = get_labevents_weights(df, max_clip)
    microbiology_weights = get_microbiology_weights(df, max_clip)
    radiology_weights = get_radiology_weights(df, max_clip)
    prescriptions_weights = get_prescriptions_weights(df, max_clip)

    return {'labevents': labevents_weights, 'microbiology': microbiology_weights,
            'radiology': radiology_weights, 'prescriptions': prescriptions_weights}


def get_positive_weights(df_path: str, max_clip: float = 50.0):
    df = pd.read_parquet(df_path)
    grouped_df = df.groupby(['subject_id', 'hadm_id', 'medical_episode'])['intervention_type'].agg(list).reset_index()
    total_episode = len(grouped_df)
    
    intervention_type_weights = get_intervention_type_weights(grouped_df, total_episode, max_clip)
    intervention_value_weights = get_intervention_value_weights(df, max_clip)

    np.savez(intervention_type_weights_path, weights=intervention_type_weights)
    np.savez(intervention_value_weights_path, weights=intervention_value_weights)


if __name__ == '__main__':
    max_clip = 500.0
    episodic_master_table_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\episodic_master_table.parquet"
    get_positive_weights(episodic_master_table_path, max_clip)
