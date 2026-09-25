import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def stratified_splits(path: str, split_ratio: list[float]) -> None:
    """
        Split the episodic master dataset into train, val and test set with Multi label stratified on DRG codes
    :param path: path to episodic master table
    :param split_ratio: split ratio for train/va/test set, Default is [0.7, 0.15, 0.15]
    :return: None
    """

    df = pd.read_parquet(path)

    patient_drg = (
        df[['subject_id', 'drg_code']]
        .drop_duplicates()
    )

    drg_matrix = (
        pd.crosstab(
            patient_drg['subject_id'],
            patient_drg['drg_code']
        )
        .astype(int)
    )
    subjects = drg_matrix.index.to_numpy()

    mortality = (
        df.groupby('subject_id')['mortality_24h']
        .max()
        .reindex(subjects, fill_value=0)
        .to_numpy()
        .reshape(-1, 1)
        .astype(int)
    )

    time_bins = [
        '<=6h',
        '6-12h',
        '12-24h',
        '24-36h',
        '>36h'
    ]

    time_matrix = (
        pd.crosstab(
            df['subject_id'],
            df['time_to_next_episode']
        )
        .reindex(columns=time_bins, fill_value=0)
        .reindex(index=subjects, fill_value=0)
        .astype(int)
    )

    X = np.zeros((len(subjects), 1))
    Y = np.concatenate(
        [
            drg_matrix.to_numpy(),
            mortality,
            time_matrix.to_numpy()
        ],
        axis=1
    )

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=split_ratio[2],
        random_state=28
    )

    train_val_idx, test_idx = next(
        msss.split(X, Y)
    )

    train_val_subjects = subjects[train_val_idx]
    test_subjects = subjects[test_idx]

    print(len(train_val_subjects), len(test_subjects), bool(set(train_val_subjects) & set(test_subjects)))

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=split_ratio[1],
        random_state=28
    )

    train_idx, val_idx = next(
        msss.split(
            np.zeros((len(train_val_subjects), 1)),
            Y[train_val_idx]
        )
    )

    train_subjects = train_val_subjects[train_idx]
    val_subjects = train_val_subjects[val_idx]

    print(len(train_subjects), len(val_subjects), bool(set(train_subjects) & set(val_subjects)))

    train_subjects_df = pd.DataFrame(train_subjects, columns=['subject_id'])
    val_subjects_df = pd.DataFrame(val_subjects, columns=['subject_id'])
    test_subjects_df = pd.DataFrame(test_subjects, columns=['subject_id'])

    train_subjects_df.to_csv(r"D:\data\Project Medical AI\datasets_v2.0\train_subjects.csv", index=False)
    val_subjects_df.to_csv(r"D:\data\Project Medical AI\datasets_v2.0\val_subjects.csv", index=False)
    test_subjects_df.to_csv(r"D:\data\Project Medical AI\datasets_v2.0\test_subjects.csv", index=False)

    print(
        f'Train subjects: {len(train_subjects_df)}, val subjects: {len(val_subjects_df)}, test subjects: {len(test_subjects_df)}')


if __name__ == '__main__':
    train_val_test_ratio = [0.7, 0.15, 0.15]
    episodic_master_table_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\episodic_master_table.parquet"
    stratified_splits(episodic_master_table_path, train_val_test_ratio)
