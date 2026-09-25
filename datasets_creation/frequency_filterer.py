import pandas as pd

from datasets_creation.frequency_summarizer import prescription_freq_summarizer, lab_freq_summarizer, \
    micro_freq_summarizer, radiology_freq_summarizer


def lab_filter(df: pd.DataFrame, freq: int=500):
    # 500 for 47k, 100 for 10k
    lab_freq_df = lab_freq_summarizer()
    lab_freq_df = lab_freq_df[lab_freq_df['frequency'] >= freq]

    valid_intervention_lists = lab_freq_df['intervention_value'].unique()
    print(f'Before filtering  labevents count: {len(df)}')
    df = df[df['itemid'].isin(valid_intervention_lists)]
    print(f'After filtering labevents count: {len(df)}')

    return df


def microbiology_filter(df: pd.DataFrame, freq: int=500):
    # 500 for 47k, 100 for 10k
    micro_freq_df = micro_freq_summarizer()
    micro_freq_df = micro_freq_df[micro_freq_df['frequency'] >= freq]

    valid_intervention_lists = micro_freq_df['intervention_value'].unique()
    print(f'Before filtering  microbiologyevents count: {len(df)}')
    df = df[df['test_itemid'].isin(valid_intervention_lists)]
    print(f'After filtering microbiologyevents count: {len(df)}')

    return df


def prescriptions_filter(df: pd.DataFrame, freq: int=500):
    # 500 for 47k, 100 for 10k
    pres_freq_df = prescription_freq_summarizer()
    pres_freq_df = pres_freq_df[pres_freq_df['frequency'] >= freq]

    valid_intervention_lists = pres_freq_df['intervention_value'].unique()

    print(f'Before filtering prescriptions count: {len(df)}')
    df = df[df['drug'].isin(valid_intervention_lists)]
    print(f'After filtering prescriptions count: {len(df)}')

    return df



def radiology_filter(df, freq: int=500):
    # 500 for 47k, 100 for 10k
    radio_freq_df = radiology_freq_summarizer()
    radio_freq_df = radio_freq_df[radio_freq_df['frequency'] >= freq]

    valid_intervention_lists = radio_freq_df['intervention_value'].unique()
    print(f"Before filtering radiology count: {len(df)}")

    def filter_exam_codes(codes):
        if pd.isna(codes):
            return codes

        codes = [code.strip() for code in codes.split(",")]

        valid_codes = [
            code for code in codes
            if code in valid_intervention_lists
        ]

        return ",".join(valid_codes) if valid_codes else None

    df["exam_code"] = df["exam_code"].apply(filter_exam_codes)

    df = df[df["exam_code"].notna()].copy()
    print(f"After filtering radiology count: {len(df)}")

    return df

if __name__ == '__main__':
    pass
