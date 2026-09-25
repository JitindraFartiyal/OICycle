import pandas as pd


def get_cohort_table(save_path: str, min_threshold: int, cohort_size: int=-1) -> None:
    patients = pd.read_parquet(r'D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\patients.parquet')
    admissions = pd.read_parquet(r'D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\admissions.parquet')
    discharge = pd.read_parquet(r'D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\discharge.parquet')

    # Remove multiple entries for a single truth source for admittime and notes charttime
    admissions = admissions.sort_values(by=['subject_id', 'admittime']).drop_duplicates(subset='subject_id', keep='first').reset_index(drop=True)
    discharge = discharge.sort_values(['subject_id', 'hadm_id', 'charttime']).drop_duplicates(subset=['subject_id'], keep='first').reset_index(drop=True)


    admissions.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\cohort\reduced_admissions.parquet", compression='zstd')
    discharge.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\cohort\reduced_notes.parquet", compression='zstd')

    # Remove subjects for which notes are empty or does not contain any text or number
    discharge = discharge.dropna(axis=0, how='any', ignore_index=False)
    alphanumeric_mask = discharge['text'].str.contains(r'[a-zA-Z0-9]', regex=True)
    discharge = discharge[alphanumeric_mask]

    cohort = patients.merge(
        admissions,
        on='subject_id',
        how='inner',
        validate='one_to_one',
    ).merge(
        discharge,
        on=['subject_id', 'hadm_id'],
        how='inner',
        validate='one_to_one',
    )[['subject_id', 'hadm_id', 'note_id']]

    # Remove subjects for which the DRG count is less than the minimum threshold
    # cohort = remove_low_freq_drg_cohort(cohort, min_threshold=min_threshold)

    cohort = cohort.astype({'subject_id': 'str', 'hadm_id': 'str', 'note_id': 'str'}, errors='raise')
    if cohort_size > 0:
        cohort = cohort.sample(n=cohort_size)
    cohort = cohort.reset_index(drop=True)

    print(f'Total cohort patients: {len(cohort)} and unique patients: {cohort['subject_id'].nunique()}')
    cohort.to_parquet(save_path, compression='zstd')


def remove_low_freq_drg_cohort(cohort_df: pd.DataFrame, min_threshold: int = 100) -> pd.DataFrame:
    drg_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\drgcodes.parquet")
    drg_df = drg_df.dropna(subset='drg_code')
    drg_df = drg_df.drop_duplicates(subset=['subject_id', 'hadm_id', 'drg_code'])

    # Drop cohort for whose we do not have drg code
    columns = cohort_df.columns.tolist() + ['drg_code']
    cohort_df = cohort_df.merge(
        drg_df,
        on=['subject_id', 'hadm_id'],
        how='left',
        validate='one_to_many',
    )[columns]
    cohort_df = cohort_df.dropna(subset=['drg_code'])

    # Remove low freq drg codes and respective subjects
    drg_code_count = drg_df.groupby(by=['drg_code'])['subject_id'].count()
    non_repr_drg_code = drg_code_count[drg_code_count < min_threshold]
    unique_repr_subjects = drg_df[~drg_df['drg_code'].isin(non_repr_drg_code.index)]['subject_id'].unique()
    cohort_df = cohort_df[cohort_df['subject_id'].isin(unique_repr_subjects)]

    cohort_df = cohort_df.drop_duplicates(subset=['subject_id', 'hadm_id', 'note_id'])[['subject_id', 'hadm_id', 'note_id']]

    return cohort_df


if __name__ == '__main__':
    min_diagnosis_threshold = 10
    cohort_size = -1
    filepath = r'D:\data\Project Medical AI\datasets_v2.0\cohort\cohort.parquet'
    get_cohort_table(filepath, min_diagnosis_threshold, cohort_size)