import pandas as pd

def prescription_freq_summarizer():
    prescription_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_prescriptions_10k.parquet")
    _df = prescription_df.copy()
    prescription_freq = {}

    for drug in _df["drug"].dropna().unique():
        prescription_freq[drug] = (
            _df[_df["drug"] == drug]["subject_id"]
            .nunique()
        )

    prescription_freq = (
        pd.Series(prescription_freq, name="frequency")
        .sort_values(ascending=False)
    )
    prescription_freq = pd.DataFrame(prescription_freq).reset_index(drop=False)
    prescription_freq = prescription_freq.rename(columns={'index': 'intervention_value'})

    return prescription_freq


def lab_freq_summarizer():
    lab_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_labevents_10k.parquet")
    _df = lab_df.copy()
    lab_freq = {}

    for item in _df["itemid"].dropna().unique():
        lab_freq[item] = (
            _df[_df["itemid"] == item]["subject_id"]
            .nunique()
        )

    lab_freq = (
        pd.Series(lab_freq, name="frequency")
        .sort_values(ascending=False)
    )
    lab_freq = pd.DataFrame(lab_freq).reset_index(drop=False)
    lab_freq = lab_freq.rename(columns={'index': 'intervention_value'})

    return lab_freq

def micro_freq_summarizer():
    micro_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_microbiologyevents_10k.parquet")
    _df = micro_df.copy()
    micro_freq = {}

    for test_item in _df["test_itemid"].dropna().unique():
        micro_freq[test_item] = (
            _df[_df["test_itemid"] == test_item]["subject_id"]
            .nunique()
        )

    micro_freq = (
        pd.Series(micro_freq, name="frequency")
        .sort_values(ascending=False)
    )
    micro_freq = pd.DataFrame(micro_freq).reset_index(drop=False)
    micro_freq = micro_freq.rename(columns={'index': 'intervention_value'})

    return micro_freq

def radiology_freq_summarizer():
    radio_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_radiology_10k.parquet")
    _df = radio_df.copy()
    _df["exam_code"] = _df["exam_code"].str.split(",")
    _df = _df.explode("exam_code", ignore_index=True)

    radio_freq = {}

    for ex_code in _df["exam_code"].dropna().unique():
        radio_freq[ex_code] = (_df[_df["exam_code"] == ex_code]["subject_id"].nunique())

    radio_freq = (pd.Series(radio_freq, name="frequency").sort_values(ascending=False))

    radio_freq = pd.DataFrame(radio_freq).reset_index(drop=False)
    radio_freq = radio_freq.rename(columns={'index': 'intervention_value'})

    return radio_freq


if __name__ == "__main__":
    pass