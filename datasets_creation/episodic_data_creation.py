import os
import numpy as np
import pandas as pd

from datasets_creation.inconsistencies import incorrect_data_subject_ids


def merge_radiology_cxr(img_path: os.path|str, radiology_df: pd.DataFrame, cxr_df: pd.DataFrame, merging_window: float=1.0) -> None:
    radiology_df['charttime'] = pd.to_datetime(radiology_df['charttime'])
    cxr_df['charttime'] = pd.to_datetime(cxr_df['charttime'])

    radiology_df = radiology_df.sort_values(by='charttime', ascending=True)
    cxr_df = cxr_df.sort_values(by='charttime')

    merged_df = pd.merge_asof(
        radiology_df,
        cxr_df,
        on='charttime',
        by=['subject_id', 'hadm_id', 'note_id'],
        direction='nearest',
        allow_exact_matches=False,
        tolerance=pd.Timedelta(hours=merging_window),
        suffixes=("_rad", "_cxr")
    ).reset_index(drop=True)

    merged_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_merged_radiology.parquet", compression='zstd')


def add_notes_data(master_table: pd.DataFrame) -> pd.DataFrame:
    preprocessed_notes = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_notes.parquet")

    print(f'Before adding notes size of master table: {len(master_table)} and unique patients: {master_table['subject_id'].nunique()}')
    new_notes_df = pd.DataFrame({
        'subject_id': preprocessed_notes['subject_id'].astype(int),
        'hadm_id': preprocessed_notes['hadm_id'].astype(int),
        'medical_intervention': 'DI',
        'medical_episode': np.nan,
        'intervention_datetime': np.nan,
        'intervention_type': 'notes',
        'intervention_value': preprocessed_notes[['demographics_embed_idx', 'allergies_embed_idx', 'chief_complaint_embed_idx']].
        apply(lambda x: ','.join(str(v) for v in x if pd.notna(v)), axis=1),
    })

    updated_table = pd.concat([master_table, new_notes_df], ignore_index=True)
    print(f'After adding notes size of master table: {len(updated_table)} and unique patients: {updated_table['subject_id'].nunique()}')

    return updated_table


def add_labevents_data(master_table: pd.DataFrame) -> pd.DataFrame:
    preprocessed_labevents = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_labevents.parquet")

    print(f'Before adding labevents size of master table: {len(master_table)} and unique patients: {master_table['subject_id'].nunique()}')
    new_labevents_df = pd.DataFrame({
        'subject_id': preprocessed_labevents['subject_id'].astype(int),
        'hadm_id': preprocessed_labevents['hadm_id'].astype(int),
        'medical_intervention': 'DI',
        'medical_episode': np.nan,
        'intervention_datetime': preprocessed_labevents['charttime'].astype(str),
        'intervention_type': 'labevents',
        'intervention_value': preprocessed_labevents[['itemid', 'value']].
        apply(lambda x: ','.join(str(v) for v in x if pd.notna(v)), axis=1),
    })

    updated_table = pd.concat([master_table, new_labevents_df], ignore_index=True)
    print(f'After adding labevents size of master table: {len(updated_table)} and unique patients: {updated_table['subject_id'].nunique()}')

    return updated_table


def add_microbiologyevents_data(master_table: pd.DataFrame) -> pd.DataFrame:
    preprocessed_microbiology = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_microbiologyevents.parquet")

    print(f'Before adding microbiology events size of master table: {len(master_table)} and unique patients: {master_table['subject_id'].nunique()}')
    new_microbiology_df = pd.DataFrame({
        'subject_id': preprocessed_microbiology['subject_id'].astype(int),
        'hadm_id': preprocessed_microbiology['hadm_id'].astype(int),
        'medical_intervention': 'DI',
        'medical_episode': np.nan,
        'intervention_datetime': preprocessed_microbiology['charttime'].astype(str),
        'intervention_type': 'microbiologyevents',
        'intervention_value': preprocessed_microbiology[['test_itemid', 'text_embed_idx']].
        apply(lambda x: ','.join(str(v) for v in x if pd.notna(v)), axis=1),
    })

    updated_table = pd.concat([master_table, new_microbiology_df], ignore_index=True)
    print(f'After adding microbiology events size of master table: {len(updated_table)} and unique patients: {updated_table['subject_id'].nunique()}')

    return updated_table


def add_prescriptions(master_table: pd.DataFrame) -> pd.DataFrame:
    preprocessed_prescriptions = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_prescriptions.parquet")

    print(f'Before adding prescription size of master table: {len(master_table)} and unique patients: {master_table['subject_id'].nunique()}')
    new_prescriptions_df = pd.DataFrame({
        'subject_id': preprocessed_prescriptions['subject_id'].astype(int),
        'hadm_id': preprocessed_prescriptions['hadm_id'].astype(int),
        'medical_intervention': 'TI',
        'medical_episode': np.nan,
        'intervention_datetime': preprocessed_prescriptions['starttime'].astype(str),
        'intervention_type': 'prescriptions',
        'intervention_value': preprocessed_prescriptions['drug'],
    })

    updated_table = pd.concat([master_table, new_prescriptions_df], ignore_index=True)
    print(f'After adding prescriptions size of master table: {len(updated_table)} and unique patients: {updated_table['subject_id'].nunique()}')

    return updated_table


def add_merged_radiology(master_table: pd.DataFrame) -> pd.DataFrame:
    preprocessed_radiology = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_merged_radiology.parquet")
    preprocessed_radiology = preprocessed_radiology.astype({'dicom_embed_idx': 'Int64'})
    print(f'Before adding radiology size of master table: {len(master_table)} and unique patients: {master_table['subject_id'].nunique()}')
    new_radiology_df = pd.DataFrame({
        'subject_id': preprocessed_radiology['subject_id'].astype(int),
        'hadm_id': preprocessed_radiology['hadm_id'].astype(int),
        'medical_intervention': 'DI',
        'medical_episode': np.nan,
        'intervention_datetime': preprocessed_radiology['charttime'].astype(str),
        'intervention_type': 'radiology',
        'intervention_value': preprocessed_radiology[['text_embed_idx', 'exam_code', 'dicom_embed_idx']].
        apply(lambda x: ','.join(str(v) for v in x if pd.notna(v)), axis=1)
    })

    updated_table = pd.concat([master_table, new_radiology_df], ignore_index=True)
    print(f'After adding radiology size of master table: {len(updated_table)} and unique patients: {updated_table['subject_id'].nunique()}')

    return updated_table


def create_master_table() -> pd.DataFrame:
    master_table = pd.DataFrame({
        'subject_id': [],
        'hadm_id': [],
        'medical_intervention': [],
        'medical_episode': [],
        'intervention_datetime': [],
        'intervention_type': [],
        'intervention_value': [],
    })

    master_table = add_notes_data(master_table)
    master_table = add_labevents_data(master_table)
    master_table = add_microbiologyevents_data(master_table)
    master_table = add_prescriptions(master_table)
    master_table = add_merged_radiology(master_table)

    master_table = master_table.reset_index(drop=True)

    print(f'\nFinal master table size: {len(master_table)} and unique patients: {master_table['subject_id'].nunique()}')
    master_table.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\master_table.parquet", compression='zstd', index=False)
    return master_table



def add_deathtime_data(episodic_df: pd.DataFrame) -> pd.DataFrame:
    admission_df = pd.read_parquet(
        r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_admissions.parquet"
    )

    df = episodic_df.copy()

    # Convert timestamps
    df["intervention_datetime"] = pd.to_datetime(
        df["intervention_datetime"], errors="coerce"
    )
    admission_df["deathtime"] = pd.to_datetime(
        admission_df["deathtime"], errors="coerce"
    )

    # Add death time
    death_df = (
        admission_df[["subject_id", "hadm_id", "deathtime"]]
        .drop_duplicates(["subject_id", "hadm_id"])
    )

    df = df.merge(
        death_df,
        on=["subject_id", "hadm_id"],
        how="left"
    )

    # Episode start/end
    episode_times = (
        df.groupby(["subject_id", "hadm_id", "medical_episode"])
        ["intervention_datetime"]
        .agg(
            episode_start="min",
            episode_end="max"
        )
        .reset_index()
    )

    df = df.merge(
        episode_times,
        on=["subject_id", "hadm_id", "medical_episode"],
        how="left"
    )

    # Default: no death within next 24h
    df["mortality_24h"] = 0

    # Death occurs during the current episode
    death_in_episode = (
        df["deathtime"].notna()
        & (df["deathtime"] >= df["episode_start"])
        & (df["deathtime"] <= df["episode_end"])
    )

    # Death occurs after the current episode but within 24h
    death_after_episode = (
        df["deathtime"].notna()
        & (df["deathtime"] > df["episode_end"])
        & (
            df["deathtime"] <=
            df["episode_end"] + pd.Timedelta(hours=24)
        )
    )

    # Do not use episodes that already contain death
    df.loc[death_in_episode, "mortality_24h"] = -1

    # Valid mortality prediction target
    df.loc[death_after_episode, "mortality_24h"] = 1

    return df

def add_next_clinical_intervention_timeperiod(episodic_df: pd.DataFrame) -> pd.DataFrame:
    bins=(6, 12, 24, 36),
    target_col="time_to_next_episode"
    df = episodic_df.copy()

    df["intervention_datetime"] = pd.to_datetime(df["intervention_datetime"], errors="coerce")

    episode_times = (
        df[["subject_id", "hadm_id", "medical_episode", "intervention_datetime"]]
        .drop_duplicates(subset=["subject_id", "hadm_id", "medical_episode"])
        .sort_values(["subject_id", "hadm_id", "intervention_datetime"])
    )

    episode_times["next_episode_datetime"] = episode_times.groupby(["subject_id", "hadm_id"])["intervention_datetime"].shift(-1)

    episode_times["hours_to_next_episode"] = (
                                                     episode_times["next_episode_datetime"]
                                                     - episode_times["intervention_datetime"]
                                             ).dt.total_seconds() / 3600

    conditions = [
        episode_times["hours_to_next_episode"].le(6),
        episode_times["hours_to_next_episode"].gt(6)
        & episode_times["hours_to_next_episode"].le(12),
        episode_times["hours_to_next_episode"].gt(12)
        & episode_times["hours_to_next_episode"].le(24),
        episode_times["hours_to_next_episode"].gt(24)
        & episode_times["hours_to_next_episode"].le(36),
        episode_times["hours_to_next_episode"].gt(36),
    ]

    choices = [
        "<=6h",
        "6-12h",
        "12-24h",
        "24-36h",
        ">36h",
    ]

    episode_times[target_col] = np.select(
        conditions,
        choices,
        default="no_next_episode"
    )

    df = df.merge(
        episode_times[["subject_id", "hadm_id", "medical_episode", "next_episode_datetime", "hours_to_next_episode", target_col]],
        on=["subject_id", "hadm_id", "medical_episode"],
        how="left"
    )

    return df


def assign_medical_episode(master_df: pd.DataFrame) -> pd.DataFrame:
    master_df = master_df.copy()

    # Detect episode changes within each patient admission
    new_episode = (
        (master_df['medical_intervention'] !=
         master_df.groupby(['subject_id', 'hadm_id'])['medical_intervention'].shift())
    )

    # Assign episode number starting from 0 for each subject/hadm
    master_df['medical_episode'] = (
        new_episode
        .groupby([master_df['subject_id'], master_df['hadm_id']])
        .cumsum()
        - 1
    ).reset_index(drop=True)

    master_df['intervention_datetime'] = pd.to_datetime(
        master_df['intervention_datetime'],
        errors='coerce'
    )

    ti_window = pd.Timedelta(hours=6)
    di_window = pd.Timedelta(hours=72)

    previous_time = master_df.groupby(
        ['subject_id', 'hadm_id']
    )['intervention_datetime'].shift()

    previous_intervention = master_df.groupby(
        ['subject_id', 'hadm_id']
    )['medical_intervention'].shift()

    time_gap = master_df['intervention_datetime'] - previous_time

    window = master_df['medical_intervention'].map({
        'TI': ti_window,
        'DI': di_window
    })

    temporal_break = time_gap > window
    new_temporal_episode = (
            (master_df['medical_intervention'] != previous_intervention)
            | temporal_break
    )

    master_df['medical_episode'] = (
            new_temporal_episode
            .groupby([master_df['subject_id'], master_df['hadm_id']])
            .cumsum()
            - 1
    ).reset_index(drop=True)

    return master_df


def remove_duplicated_prescriptions(df: pd.DataFrame) -> pd.DataFrame:
    prescription_mask = df["intervention_type"].eq("prescriptions")

    df_prescription = df[prescription_mask].drop_duplicates(
        subset=["subject_id", "medical_episode", "intervention_value"]
    )

    df_non_prescription = df[~prescription_mask]

    df = (
        pd.concat([df_non_prescription, df_prescription])
        .sort_index()
        .reset_index(drop=True)
    )
    return df

def create_episodic_master_table(master_df: pd.DataFrame) -> None:
    master_df = (
        master_df
        .sort_values(
            by=["subject_id", "intervention_datetime"],
            ascending=[True, True],
            na_position="first"
        )
        .reset_index(drop=True)
    )

    episodic_master_df = assign_medical_episode(master_df)
    episodic_master_df = episodic_master_df.astype({'subject_id': 'Int64', 'hadm_id': 'Int64'})

    # Remove patients which have more than episode length of 50
    episode_counts = episodic_master_df.groupby('subject_id')['medical_episode'].nunique()
    episodic_master_df = episodic_master_df[
        episodic_master_df['subject_id'].isin(episode_counts[episode_counts < 50].index)
    ]

    # Remove incorrect data subjects
    episodic_master_df = episodic_master_df[~episodic_master_df['subject_id'].isin(incorrect_data_subject_ids)]

    # Add DRG codes
    drg_icd_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\drgcodes.parquet")
    drg_icd_df = drg_icd_df.astype({'drg_code': 'Int64'})

    aggregated_drg = (
        drg_icd_df
        .groupby(['subject_id', 'hadm_id'])['drg_code']
        .agg(lambda x: ','.join(str(v) for v in x.dropna()))
        .reset_index()
    )

    aggregated_drg = aggregated_drg.astype({'subject_id': 'str', 'hadm_id': 'str'})
    episodic_master_df = episodic_master_df.astype({'subject_id': 'str', 'hadm_id': 'str'})
    episodic_master_df_cols = episodic_master_df.columns.tolist()
    episodic_master_df_cols += ['drg_code']
    episodic_master_df = episodic_master_df.merge(
        aggregated_drg,
        on=['subject_id', 'hadm_id'],
        how='left',
    )[episodic_master_df_cols]
    episodic_master_df = add_deathtime_data(episodic_master_df)
    episodic_master_df = add_next_clinical_intervention_timeperiod(episodic_master_df)
    episodic_master_df = remove_duplicated_prescriptions(episodic_master_df)
    episodic_master_df = episodic_master_df.dropna(subset=['drg_code'])
    episodic_master_df = episodic_master_df.reset_index(drop=True)

    print(f'Episodic master table size: {len(episodic_master_df)} and unique patients: {episodic_master_df['subject_id'].nunique()}')
    episodic_master_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\episodic_master_table.parquet",
                              compression='zstd', index=False)


if __name__ == "__main__":
    cxr_scans_path = r"D:\data\Project Medical AI\datasets_v2.0\cxr"
    radiology_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_radiology.parquet")
    cxr_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_cxr.parquet")

    radiology_cxr_merge_window = 1.0
    merge_radiology_cxr(cxr_scans_path, radiology_df, cxr_df, radiology_cxr_merge_window)
    master_table = create_master_table()

    # master_table = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\master_table.parquet")
    create_episodic_master_table(master_table)