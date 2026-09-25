import os
import pandas as pd


def create_cohort_measurements(name: str, measurement_path: str, columns: list[str], save_path: str, cohort: pd.DataFrame) -> None:
    df = pd.read_parquet(measurement_path, columns=columns)

    df = df.astype({'subject_id': 'str'})

    if name in ['patients', 'cxr']:
        cohort_measurement_df = cohort.merge(df, on='subject_id', how='left')
    else:
        df = df.astype({'hadm_id': 'str'})
        cohort_measurement_df = cohort.merge(df, on=['subject_id', 'hadm_id'], how='left')

    if name == 'radiology':
        cohort_measurement_df = cohort_measurement_df.rename(columns={'note_id_x': 'note_id', 'note_id_y': 'note_radiology'})

    if name == 'cxr':
        cohort_measurement_df = cohort_measurement_df.astype({'study_id': 'str'})

    cohort_measurement_df = cohort_measurement_df.astype({'subject_id': 'str', 'hadm_id': 'str', 'note_id': 'str'}, errors='raise').reset_index(drop=True)

    print(f'Size of cohort measurements for {name}: {len(cohort_measurement_df)} and unique patients: {cohort_measurement_df['subject_id'].nunique()}')
    cohort_measurement_df.to_parquet(save_path, compression='zstd')

if __name__ == '__main__':
    save_path = r"D:\data\Project Medical AI\datasets_v2.0\cohort"
    cohort_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort.parquet")
    print(f'Size of cohort {len(cohort_df)} and unique patients: {cohort_df['subject_id'].nunique()}')

    cohort_measurements = {
        'patients': {
            'path': r"D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\patients.parquet",
            'columns': ['subject_id', 'gender', 'anchor_age'],
        },
        'admissions':{
            'path': r"D:\data\Project Medical AI\datasets_v2.0\cohort\reduced_admissions.parquet",
            'columns': ['subject_id', 'hadm_id', 'deathtime'],
        },
        'notes': {
            'path': r"D:\data\Project Medical AI\datasets_v2.0\cohort\reduced_notes.parquet",
            'columns': ['subject_id', 'hadm_id', 'charttime', 'text'],
        },
        'labevents': {
            'path': r"D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\labevents.parquet",
            'columns': ['subject_id', 'hadm_id', 'charttime', 'itemid', 'value', 'ref_range_lower', 'ref_range_upper'],
        },
        'microbiologyevents': {
            'path': r"D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\microbiologyevents.parquet",
            'columns': ['subject_id', 'hadm_id', 'charttime', 'test_name', 'test_itemid', 'comments'],
        },
        'prescriptions': {
            'path': r"D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\prescriptions.parquet",
            'columns': ['subject_id', 'hadm_id', 'starttime', 'drug'],
        },
        'radiology':{
            'path': r"D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\radiology.parquet",
            'columns': ['subject_id', 'hadm_id', 'charttime', 'note_id', 'text']
        },
        'cxr':{
            'path': r"D:\data\Project Medical AI\datasets\mimic-cxr-jpg-2.1.0\mimic-cxr-2.0.0-metadata.parquet",
            'columns': ['dicom_id', 'subject_id', 'study_id', 'StudyDate', 'StudyTime']
        }
    }

    for name, value in cohort_measurements.items():
        print(f'Creating cohort measurements for {name}')
        create_cohort_measurements(
            name=name,
            measurement_path=value['path'],
            columns=value['columns'],
            save_path=os.path.join(save_path, f'cohort_{name}.parquet'),
            cohort=cohort_df
        )