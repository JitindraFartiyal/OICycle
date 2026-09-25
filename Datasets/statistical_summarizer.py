import re
import pandas as pd


def labevents_summarizer(path: str) -> None:
    """
        Summarize the labevents table into mean, std, and save it in a csv file
    :param path: episodic master table path
    :return: None
    """
    df = pd.read_parquet(path)
    labevents_df = df[df['intervention_type'] == 'labevents']
    labevents_df[['itemid', 'value']] = labevents_df['intervention_value'].str.split(',', n=1, expand=True)
    labevents_df = labevents_df[['itemid', 'value']].drop_duplicates(subset='itemid', keep='first')

    ranges_labevents_df = pd.read_parquet(
        r"D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\labevents.parquet",
        columns=[
            'itemid',
            'ref_range_lower',
            'ref_range_upper',
        ])
    ranges_labevents_df = ranges_labevents_df.dropna()
    ranges_labevents_df = ranges_labevents_df.drop_duplicates(subset='itemid', keep='first')
    merged_labevents_ranges = labevents_df.merge(ranges_labevents_df, on='itemid', how='left')
    merged_labevents_ranges['index'] = range(merged_labevents_ranges.shape[0])
    print(f'Total labevents: {len(merged_labevents_ranges)}')

    merged_labevents_ranges.to_parquet(r'labevents_stats.parquet', compression='zstd')


def microbiologyevents_summarizer(path: str) -> None:
    """
        Get the microbiology events indexes
    :param path: episodic master table path
    :return: None
    """
    df = pd.read_parquet(path)
    microbiologyevents_df = df[df['intervention_type'] == 'microbiologyevents']

    microbiologyevents_df[['itemid', 'value']] = microbiologyevents_df['intervention_value'].str.split(',', n=1, expand=True)
    microbiologyevents_stats= microbiologyevents_df[['itemid', 'value']].drop_duplicates(subset='itemid', keep='first')

    microbiologyevents_stats['index'] = range(microbiologyevents_stats.shape[0])

    print(f'Total microbiologyevents: {len(microbiologyevents_stats)}')
    microbiologyevents_stats.to_parquet(r'microbiology_stats.parquet', compression='zstd')


def prescriptions_summarizer(path: str) -> None:
    """
        Get the prescriptions indexes
    :param path: episodic master table path
    :return: None
    """
    df = pd.read_parquet(path)
    prescriptions_df = df[df['intervention_type'] == 'prescriptions']
    prescriptions_stats = pd.DataFrame(prescriptions_df['intervention_value'].unique(), columns=['itemid'])

    prescriptions_stats['index'] = range(prescriptions_stats.shape[0])

    print(f'Total prescriptions: {len(prescriptions_stats)}')
    prescriptions_stats.to_parquet(r'prescriptions_stats.parquet', compression='zstd')


def radiology_summarizer(path: str) -> None:
    """
            Get the prescriptions type
        :param path: episodic master table path
        :return: None
        """
    df = pd.read_parquet(path)
    radiology_df = df[df['intervention_type'] == 'radiology']
    radiology_exams = []
    for i, row in radiology_df.iterrows():
        intervention_value = row['intervention_value']
        for _val in intervention_value.split(','):
            char_contain = bool(re.search(r'[a-zA-Z]', _val))
            if char_contain and _val not in radiology_exams:
                radiology_exams.append(_val)

    radiology_stats = pd.DataFrame(radiology_exams, columns=['itemid'])
    radiology_stats['index'] = range(radiology_stats.shape[0])

    print(f'Total radiology exams: {len(radiology_stats)}')
    radiology_stats.to_parquet(r'radiology_stats.parquet', compression='zstd')

if __name__ == '__main__':
    episodic_master_table_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\episodic_master_table.parquet"

    labevents_summarizer(episodic_master_table_path)
    microbiologyevents_summarizer(episodic_master_table_path)
    prescriptions_summarizer(episodic_master_table_path)
    radiology_summarizer(episodic_master_table_path)