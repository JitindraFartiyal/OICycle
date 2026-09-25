import os
import numpy as np
import pandas as pd
import torch
import random

from torch.utils.data import Dataset, DataLoader
from Datasets.utils import labevents_preprocessor, microbiologyevents_preprocessor, prescriptions_preprocessor, \
    get_ground_truth, radiology_preprocessor


class ObserverInterveneDataset(Dataset):
    def __init__(self, df_path: str, train_val_test_path: str, embedding_dirpath: str, radiology_path: str, size=-1):
        self.df = pd.read_parquet(df_path)
        self.subjects = pd.read_csv(train_val_test_path)['subject_id'].tolist()

        if size != -1:
            self.subjects = random.sample(self.subjects, size)

        self.embedding_dirpath = embedding_dirpath
        self.radiology_path = radiology_path

        self.demo_embedding = np.load(os.path.join(embedding_dirpath, 'demo_embedding.npy'))
        self.allergies_embedding = np.load(os.path.join(embedding_dirpath, 'allergies_embedding.npy'))
        self.complaint_embedding = np.load(os.path.join(embedding_dirpath, 'complaint_embedding.npy'))
        self.microbiology_embedding = np.load(os.path.join(embedding_dirpath, 'microbiology_embedding.npy'))
        self.radiology_embedding = np.load(os.path.join(embedding_dirpath, 'radiology_embedding.npy'))
        self.cxr_embedding = np.load(os.path.join(embedding_dirpath, 'cxr_embedding.npy'))

        self.time_to_next_episode_dict = {
            "<=6h": 0,
            "6-12h": 1,
            "12-24h": 2,
            "24-36h": 3,
            ">36h": 4,
        }

    def __len__(self):
        return len(self.subjects)


    def note_embedder(self, intervention_value: pd.Series):
        embed_idx = intervention_value.str.split(',').values[0]
        demo_embedding = torch.tensor(self.demo_embedding[int(embed_idx[0])])
        allergies_embedding = torch.tensor(self.allergies_embedding[int(embed_idx[1])])
        complaint_embedding = torch.tensor(self.complaint_embedding[int(embed_idx[2])])
        notes_embedding = torch.cat([demo_embedding, allergies_embedding, complaint_embedding], dim=0).unsqueeze(0)
        # print(notes_embedding.shape)

        return notes_embedding


    def labevents_embedder(self, intervention_value: pd.Series):
        lab_itemid, lab_value = labevents_preprocessor(intervention_value)
        lab_itemid = torch.tensor(lab_itemid).unsqueeze(1)
        lab_value = torch.tensor(lab_value).unsqueeze(1)
        # print(lab_itemid.shape, lab_value.shape)

        return lab_itemid, lab_value


    def microbiologyevents_embedder(self, intervention_value: pd.Series):
        microbiologyevents_id, microbiologyevents_embed_idx = microbiologyevents_preprocessor(intervention_value)
        microbiologyevents_id = torch.tensor(microbiologyevents_id).unsqueeze(1)
        microbiologyevents_embedding = []
        for _idx in microbiologyevents_embed_idx:
            microbiologyevents_embedding.append(self.microbiology_embedding[int(_idx)])
        microbiologyevents_embedding = torch.tensor(np.array(microbiologyevents_embedding))
        # print(microbiologyevents_id.shape, microbiologyevents_embedding.shape)

        return microbiologyevents_id, microbiologyevents_embedding


    def prescriptions_embedder(self, intervention_value: pd.Series):
        prescriptions_id = prescriptions_preprocessor(intervention_value)
        prescriptions_id = torch.tensor(prescriptions_id).unsqueeze(1)
        # print(prescriptions_id.shape)

        return prescriptions_id


    def radiology_embedder(self, intervention_value: pd.Series):
        embed_id, exam_codes, dicom_embed_idx = radiology_preprocessor(intervention_value)
        radiology_exam_code_embedding = torch.tensor(exam_codes).unsqueeze(1)

        if len(dicom_embed_idx) == 0:
            radiology_embedding = torch.tensor(self.radiology_embedding[embed_id]).unsqueeze(0)
        else:
            radiology_embedding = torch.tensor(self.cxr_embedding[dicom_embed_idx])

        # print(radiology_exam_code_embedding.shape, radiology_embedding.shape)

        return radiology_exam_code_embedding, radiology_embedding


    def __getitem__(self, idx):
        try:
            try:
                subject_id = str(self.subjects[idx])
                patient_data = self.df[self.df['subject_id'] == subject_id]
                max_episodes = patient_data['medical_episode'].max().item()
            except Exception as e:
                print('subject issue:', subject_id, e)
            episodical_data = []
            for episode in range(max_episodes):
                curr_episode_data= patient_data[patient_data['medical_episode'] == episode]
                time_to_next_episode = (
                    curr_episode_data[curr_episode_data['time_to_next_episode'] != 'no_next_episode']
                    .sort_values("hours_to_next_episode")["time_to_next_episode"]
                    .values
                )

                curr_episode_values = {
                    'episode': episode,
                    'prediction_state': False,
                    'prediction_type': None,
                    'target': None,
                    'time_to_next_episode_target': torch.Tensor([(self.time_to_next_episode_dict[time_to_next_episode[0]] if len(time_to_next_episode) > 0 else -1)]),
                    'mortality_target': torch.Tensor([curr_episode_data['mortality_24h'].max().item()]),
                }
                unique_interventions = curr_episode_data['intervention_type'].unique().tolist()

                per_episode_values = []
                for unique_intervention in unique_interventions:
                    curr_intervention_data = curr_episode_data[curr_episode_data['intervention_type'] == unique_intervention]
                    intervention_value = curr_intervention_data['intervention_value']
                    if unique_intervention  == 'notes':
                        intervention_embedding = self.note_embedder(intervention_value)
                    elif unique_intervention == 'labevents':
                        lab_item_embedding, lab_value_embedding = self.labevents_embedder(intervention_value)
                        intervention_embedding = (lab_item_embedding, lab_value_embedding)
                    elif unique_intervention == 'microbiologyevents':
                        microbiologyevents_id_embedding, microbiologyevents_embedding = self.microbiologyevents_embedder(intervention_value)
                        intervention_embedding = (microbiologyevents_id_embedding, microbiologyevents_embedding)
                    elif unique_intervention == 'prescriptions':
                        intervention_embedding = self.prescriptions_embedder(intervention_value)
                    elif unique_intervention == 'radiology':
                        try:
                            radiology_exam_code_embedding, radiology_embedding = self.radiology_embedder(intervention_value)
                            intervention_embedding = (radiology_exam_code_embedding, radiology_embedding)
                        except Exception as e:
                            print('Observe Intervene Dataloader: Radiology embedding issue:', e)
                    per_episode_values.append({
                        'intervention_type': unique_intervention,
                        'intervention_embedding': intervention_embedding,
                    })

                curr_episode_values['interventions'] = per_episode_values
                episodical_data.append(curr_episode_values)

                if episode > 0 and (
                    'labevents' in unique_interventions or
                    'microbiologyevents' in unique_interventions or
                    'prescriptions' in unique_interventions or
                    'radiology' in unique_interventions
                ):
                    episodical_data[-2]['prediction_state'] = True
                    if (
                            'labevents' in unique_interventions or
                            'microbiologyevents' in unique_interventions or
                            'radiology' in unique_interventions
                    ):
                        episodical_data[-2]['prediction_type'] = 'DI'
                    else:
                        episodical_data[-2]['prediction_type'] = 'TI'
        except Exception as e:
            print(e)
            print(idx, self.subjects[idx], len(self.subjects))

        gt = get_ground_truth(episodical_data)

        return {
            'subject_id': subject_id,
            'episodical_data': episodical_data,
            'gt': gt
        }


def collate_patient(batch):
    return batch


def get_dataloader(episodic_data_path: str, path: str, embedding_path: str, cxr_path: str, size=-1):
    train_ds = ObserverInterveneDataset(episodic_data_path, path, embedding_path, cxr_path, size=size)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_patient)

    return train_dl


if __name__ == '__main__':
    episodic_data_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\episodic_master_table.parquet"
    train_path = r"D:\data\Project Medical AI\datasets_v2.0\train_subjects.csv"
    embedding_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed"
    cxr_path = r"D:\data\Project Medical AI\datasets_v2.0\cxr"

    train_dl = get_dataloader(episodic_data_path, train_path, embedding_path, cxr_path)
    sample = next(iter(train_dl))[0]
    print(sample)

    # dataset = ObserverInterveneDataset(episodic_data_path, train_path, embedding_path, cxr_path, scan_size)
    # dataset[32584] #10, 11914, 32584