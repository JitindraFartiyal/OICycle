import os
import re
import numpy as np
import pandas as pd
import torch
import tqdm

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from clinical_radiology_notes_utils import remove_word, clean, notes_regex
from Networks.medical_text_embedder import MedicalTextEmbedder
from datasets_creation.clinical_radiology_notes_utils import radiology_regex
from datasets_creation.frequency_filterer import prescriptions_filter, lab_filter, microbiology_filter, radiology_filter


def embed_column(output_path: str, df: pd.DataFrame, column: str, embedder: MedicalTextEmbedder, batch_size: int=256) -> None:
    texts = df[column].fillna("").astype(str).tolist()

    embeddings = []
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        emb = embedder(batch)
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    np.save(output_path, embeddings)


def embed_medical_image(output_path: str, df: pd.DataFrame) -> list:
    embeddings = []
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    encoder = AutoModel.from_pretrained("microsoft/rad-dino")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    encoder.eval()

    loaded_subjects = []
    with torch.no_grad():
        for i, row in tqdm.tqdm(df.iterrows()):
            subject_id = row['subject_id']
            dicom_path = row['dicom_id']
            for path in dicom_path.split(','):
                if os.path.exists(path):
                    try:
                        image = Image.open(path)
                        image.load()
                        image = image.resize((256, 256))
                        inputs = processor(
                            images=image,
                            return_tensors="pt"
                        )
                        inputs = {
                            k: v.to(device)
                            for k, v in inputs.items()
                        }
                        outputs = encoder(**inputs)

                        cls_embedding = outputs.last_hidden_state[:, 0, :]
                        embeddings.append(
                            cls_embedding.cpu().numpy()
                        )
                        loaded_subjects.append(subject_id)
                    except (OSError, ValueError) as e:

                        print(f"\nSkipping corrupted image: {path}")
                        print(f"Reason: {e}")
                        continue
                else:
                    print(f'Path {path} does not exist')

    embeddings = np.concatenate(embeddings, axis=0)
    np.save(output_path, embeddings)

    return loaded_subjects


def preprocess_admissions() -> None:
    admissions_df = pd.read_parquet(
        r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort_admissions.parquet",
        columns=['subject_id', 'hadm_id', 'deathtime']
    )
    admissions_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_admissions.parquet",
                              compression='zstd')


def preprocess_clinical_notes() -> None:
    patients_df = pd.read_parquet(
        r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort_patients.parquet",
        columns=['subject_id', 'hadm_id', 'gender', 'anchor_age']
    )
    notes_df = pd.read_parquet(
        r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort_notes.parquet",
        columns=['subject_id', 'note_id', 'charttime', 'text']
    )

    merged_df = notes_df.merge(
        patients_df,
        how='left',
        on=['subject_id']
    )
    init_assess = {
        'subject_id': merged_df['subject_id'],
        'hadm_id': merged_df['hadm_id'],
        'note_id': merged_df['note_id'],
        'demographics_notes': 'Gender: ' + merged_df['gender'] + ' Age: ' + merged_df['anchor_age'].astype(str)
    }
    for key, rexpr in notes_regex.items():
        value = merged_df['text'].str.extract(pat=rexpr['pat'], flags=re.DOTALL)[0]
        alphanumeric_mask = value.str.contains(r'[a-zA-Z0-9]', regex=True)  # Only include notes which has at least text of numbers
        value = value[alphanumeric_mask]
        for word in rexpr['rm_word']:
            value = remove_word(value, word)

        init_assess[key] = clean(value)

    init_assess_df = pd.DataFrame(init_assess)
    print(f'Before preprocessing clinical notes: {len(init_assess_df)} and unique patients: {init_assess_df['subject_id'].nunique()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    medical_text_embedder = MedicalTextEmbedder(device)

    demo_embedding_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\demo_embedding.npy"
    allergies_embedding_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\allergies_embedding.npy"
    complaint_embedding_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\complaint_embedding.npy"

    embed_column(demo_embedding_path, init_assess_df, 'demographics_notes', medical_text_embedder)
    embed_column(allergies_embedding_path, init_assess_df, 'Allergies', medical_text_embedder)
    embed_column(complaint_embedding_path, init_assess_df, 'Chief Complaint', medical_text_embedder)

    init_assess_df["demographics_embed_idx"] = np.arange(len(init_assess_df))
    init_assess_df["allergies_embed_idx"] = np.arange(len(init_assess_df))
    init_assess_df["chief_complaint_embed_idx"] = np.arange(len(init_assess_df))
    init_assess_df = init_assess_df.dropna(axis=0, how='any', ignore_index=False)
    init_assess_df = init_assess_df.reset_index(drop=True)

    print(f'After preprocessing clinical notes: {len(init_assess_df)} and unique patients: {init_assess_df['subject_id'].nunique()}')
    init_assess_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_notes.parquet", compression='zstd')


def preprocess_labevents() -> None:
    labevents_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort_labevents.parquet")

    labitems_df = pd.read_parquet(
        r"D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\d_labitems.parquet",
        columns=[
            'itemid',
            'label'
        ]
    )
    print(f'Before preprocessing labevents: {len(labevents_df)} and unique patients: {labevents_df['subject_id'].nunique()}')

    labevents_df = labevents_df[labevents_df['value'].str.isnumeric()] # Only include numerical lab events
    labitems_df = labitems_df.astype(dtype={'itemid': 'str'}, errors='raise')
    labevents_df = labevents_df.astype(dtype={'itemid': 'str'}, errors='raise')

    labevents_df = labevents_df.merge(
        right=labitems_df,
        on='itemid',
        how='left',
    )
    labevents_df = labevents_df.dropna(axis=0, how='any', ignore_index=False)
    labevents_df = labevents_df.reset_index(drop=True)
    print(f'After preprocessing labevents: {len(labevents_df)} and unique patients: {labevents_df['subject_id'].nunique()}')
    labevents_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_labevents.parquet", compression='zstd')


def preprocess_microbiologyevents() -> None:
    microbiologyevents_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort_microbiologyevents.parquet")

    print(f'Before preprocessing microbiologyevents: {len(microbiologyevents_df)} and unique patients: {microbiologyevents_df['subject_id'].nunique()}')

    alphanumeric_mask = microbiologyevents_df['comments'].str.contains(r'[a-zA-Z0-9]', regex=True) # Only include comments which has numbers or text
    microbiologyevents_df = microbiologyevents_df[alphanumeric_mask]
    microbiologyevents_df = microbiologyevents_df.dropna(axis=0, how='any', ignore_index=False)
    microbiologyevents_df = microbiologyevents_df.astype(dtype={'test_itemid': 'str'}, errors='raise')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    medical_text_embedder = MedicalTextEmbedder(device)

    microbiology_embedding_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\microbiology_embedding.npy"
    embed_column(microbiology_embedding_path, microbiologyevents_df, 'comments', medical_text_embedder)
    microbiologyevents_df["text_embed_idx"] = np.arange(len(microbiologyevents_df))
    microbiologyevents_df = microbiologyevents_df.reset_index(drop=True)

    print(f'After preprocessing microbiologyevents: {len(microbiologyevents_df)} and unique patients: {microbiologyevents_df['subject_id'].nunique()}')
    microbiologyevents_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_microbiologyevents.parquet", compression='zstd')


def preprocess_prescriptions() -> None:
    prescriptions_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort_prescriptions.parquet")
    print(f'Before preprocessing prescriptions: {len(prescriptions_df)} and unique patients: {prescriptions_df['subject_id'].nunique()}')

    alphanumeric_mask = prescriptions_df['drug'].str.contains(r'[a-zA-Z0-9]',regex=True)  # Only include comments which has numbers or text
    prescriptions_df = prescriptions_df[alphanumeric_mask]
    prescriptions_df = prescriptions_df.dropna(axis=0, how='any', ignore_index=False)

    prescriptions_df = prescriptions_df.reset_index(drop=True)

    print(f'After preprocessing prescriptions: {len(prescriptions_df)} and unique patients: {prescriptions_df['subject_id'].nunique()}')
    prescriptions_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_prescriptions.parquet", compression='zstd')


def preprocess_radiology() -> None:
    radiology_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort_radiology.parquet")

    radiology_detail_df = pd.read_parquet(
        r'D:\data\Project Medical AI\datasets\mimic-iv-3.1\mimic-iv-3.1\parquet_tables\radiology_detail.parquet',
        columns=[
            'note_id',
            'field_name',
            'field_value',
            'field_ordinal'
        ]
    )

    print(f'Before preprocessing radiology: {len(radiology_df)} and unique patients: {radiology_df['subject_id'].nunique()}')

    # Extract exam codes into a list for each radiology note id
    radiology_exam_codes = (
        radiology_detail_df[
            radiology_detail_df["field_name"] == "exam_code"
            ]
        .groupby("note_id")["field_value"]
        .apply(lambda x: ','.join(x))
        .reset_index()
        .rename(columns={"field_value": "exam_code"})
    )
    radiology_df = radiology_df.merge(right=radiology_exam_codes, how='left', left_on='note_radiology', right_on='note_id')

    for key, rexpr in radiology_regex.items():
        value = radiology_df['text']
        # value = radiology_df['text'].str.extract(pat=rexpr['pat'], flags=re.DOTALL)[0]
        # alphanumeric_mask = value.str.contains(r'[a-zA-Z0-9]', regex=True)  # Only include radiology notes which has numbers or text atleast
        # value = value[alphanumeric_mask]
        for word in rexpr['rm_word']:
            value = remove_word(value, word)
        radiology_df['clean_text'] = value

    radiology_df = radiology_df.dropna(axis=0, how='any', ignore_index=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    medical_text_embedder = MedicalTextEmbedder(device)

    radiology_embedding_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\radiology_embedding.npy"
    embed_column(radiology_embedding_path, radiology_df, 'text', medical_text_embedder)
    radiology_df["text_embed_idx"] = np.arange(len(radiology_df))
    radiology_df = radiology_df.drop(columns=['note_id_y'])
    radiology_df = radiology_df.rename(columns={'note_id_x': 'note_id'})
    radiology_df = radiology_df.reset_index(drop=True)

    print(f'After preprocessing radiology: {len(radiology_df)} and unique patients: {radiology_df['subject_id'].nunique()}')

    radiology_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_radiology.parquet", compression='zstd')


def preprocess_cxr() -> None:
    cxr_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort_cxr.parquet")

    print(f'Before preprocessing cxr : {len(cxr_df)} and unique patients: {cxr_df['subject_id'].nunique()}')
    cxr_df = cxr_df.dropna(axis=0, how='any', inplace=False, ignore_index=False)

    cxr_df = cxr_df.astype(dtype={'StudyDate': 'int64', 'StudyTime': 'float'}, errors='raise')

    img_path = r"D:\data\Project Medical AI\datasets_v2.0\cxr"
    cxr_df['dicom_id'] = img_path + r'\\' + cxr_df['dicom_id'] + '.jpg'

    date_str = cxr_df["StudyDate"].astype(str)
    time_str = cxr_df["StudyTime"].map(lambda x: f"{x:010.3f}")
    datetime_str = date_str + time_str

    cxr_df["charttime"] = pd.to_datetime(
        datetime_str,
        format="%Y%m%d%H%M%S.%f",
        yearfirst=True,
        errors="raise"
    )

    cxr_df = cxr_df.drop(["StudyDate", "StudyTime"], axis=1)

    agg_cxr_df = (
        cxr_df.groupby('study_id')['dicom_id']
        .agg(lambda x: ','.join(x.astype(str)))
        .reset_index()
    )
    cxr_df = cxr_df.drop_duplicates(subset='study_id', keep='first')
    cxr_df = cxr_df.drop('dicom_id', axis=1)
    cxr_df = cxr_df.merge(right=agg_cxr_df, how='left', on='study_id')

    cxr_embedding_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\cxr_embedding.npy"
    loaded_subjects = embed_medical_image(cxr_embedding_path, cxr_df)
    cxr_df = cxr_df[cxr_df['subject_id'].isin(loaded_subjects)].copy()

    subject_to_embed_idx = {
        subject_id: idx
        for idx, subject_id in enumerate(loaded_subjects)
    }

    cxr_df["dicom_embed_idx"] = cxr_df["subject_id"].map(subject_to_embed_idx)

    cxr_df = cxr_df.reset_index(drop=True)

    print(f'After preprocessing cxr : {len(cxr_df)} and unique patients: {cxr_df['subject_id'].nunique()}')
    cxr_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_cxr.parquet", compression='zstd')


def clean_intermediate_tables(cohort_df: pd.DataFrame) -> None:
    im_preprocessed_notes = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_notes.parquet")
    im_preprocessed_labevents = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_labevents.parquet")
    im_preprocessed_microbiologyevents = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_microbiologyevents.parquet")
    im_preprocessed_prescriptions = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_prescriptions.parquet")
    im_preprocessed_radiology = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_radiology.parquet")
    im_preprocessed_cxr = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_cxr.parquet")
    im_preprocessed_admissions = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\im_preprocessed_admissions.parquet")

    unique_subjects = im_preprocessed_notes['subject_id'].unique().tolist()

    notes_subjects = set(unique_subjects)

    other_subjects = set(
        pd.concat([
            im_preprocessed_labevents[['subject_id']],
            im_preprocessed_microbiologyevents[['subject_id']],
            im_preprocessed_prescriptions[['subject_id']],
            im_preprocessed_radiology[['subject_id']],
            im_preprocessed_cxr[['subject_id']]
        ])['subject_id'].unique()
    )

    notes_only_subjects = notes_subjects - other_subjects
    unique_subjects = list(set(unique_subjects) - set(notes_only_subjects))

    print(f'After all preprocessing unique patients count : {len(unique_subjects)}')

    preprocessed_cohort_df = cohort_df[cohort_df['subject_id'].isin(unique_subjects)]
    preprocessed_cohort_df = preprocessed_cohort_df.reset_index(drop=True)
    preprocessed_cohort_df.to_parquet(r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_cohort.parquet", compression='zstd')

    def save_preprocessed_tables(im_df: pd.DataFrame, file_name: str) -> None:
        im_df = im_df[im_df['subject_id'].isin(unique_subjects)]
        im_df = im_df.reset_index(drop=True)
        im_df.to_parquet(rf'D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_{file_name}.parquet', compression='zstd')

    save_preprocessed_tables(im_preprocessed_notes, 'notes')
    save_preprocessed_tables(im_preprocessed_labevents, 'labevents')
    save_preprocessed_tables(im_preprocessed_microbiologyevents, 'microbiologyevents')
    save_preprocessed_tables(im_preprocessed_prescriptions, 'prescriptions')
    save_preprocessed_tables(im_preprocessed_radiology, 'radiology')
    save_preprocessed_tables(im_preprocessed_cxr, 'cxr')
    save_preprocessed_tables(im_preprocessed_admissions, 'admissions')


def filtering():
    labevents_df_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_labevents.parquet"
    microbiologyevents_df_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_microbiologyevents.parquet"
    prescriptions_df_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_prescriptions.parquet"
    radiology_df_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\preprocessed_radiology.parquet"

    labevents_df = pd.read_parquet(labevents_df_path)
    microbiologyevents_df = pd.read_parquet(microbiologyevents_df_path)
    prescriptions_df = pd.read_parquet(prescriptions_df_path)
    radiology_df = pd.read_parquet(radiology_df_path)

    labevents_df = lab_filter(labevents_df)
    microbiologyevents_df = microbiology_filter(microbiologyevents_df)
    prescriptions_df = prescriptions_filter(prescriptions_df)
    radiology_df = radiology_filter(radiology_df)

    labevents_df.to_parquet(labevents_df_path)
    microbiologyevents_df.to_parquet(microbiologyevents_df_path)
    prescriptions_df.to_parquet(prescriptions_df_path)
    radiology_df.to_parquet(radiology_df_path)

    print(f'Unique subjects after filtering in labevents: {labevents_df.subject_id.nunique()}')
    print(f'Unique subjects after filtering in microbiology: {microbiologyevents_df.subject_id.nunique()}')
    print(f'Unique subjects after filtering in prescriptions: {prescriptions_df.subject_id.nunique()}')
    print(f'Unique subjects after filtering in radiology: {radiology_df.subject_id.nunique()}')



if __name__ == '__main__':
    cohort_df = pd.read_parquet(r"D:\data\Project Medical AI\datasets_v2.0\cohort\cohort.parquet")

    preprocess_clinical_notes()
    preprocess_labevents()
    preprocess_microbiologyevents()
    preprocess_prescriptions()
    preprocess_radiology()
    preprocess_cxr()
    preprocess_admissions()

    clean_intermediate_tables(cohort_df)
    # filtering()