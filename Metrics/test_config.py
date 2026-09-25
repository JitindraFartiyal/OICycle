import pandas as pd

episodic_data_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\episodic_master_table.parquet"
embedding_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed"
cxr_path = r"D:\data\Project Medical AI\datasets_v2.0\cxr"
test_path = r"D:\data\Project Medical AI\datasets_v2.0\test_subjects.csv"
bsf = r"D:\data\Project Medical AI\code_v2.0\belief_states_latent"

model_type = 'latent' #POMDP, single_snapshot, transformers, latent

test_config = {
    'batch_size': 1,
    'hidden_dim': 512,
    'lab_embed_dim': 128,
    'embed_dim': 512,
    'threshold': 0.5,
    'resume': False,
    'load_state_model_path': r"D:\data\Project Medical AI\code_v2.0\ckpts_latent\state_model_2.pth",
    'load_episode_embedding_model_path': r"D:\data\Project Medical AI\code_v2.0\ckpts_latent\episode_embedding_model_2.pth",
    'load_init_state_path': r"D:\data\Project Medical AI\code_v2.0\ckpts_latent\init_state_2.pth"
}
# 6 for POMDP, 1 for single-snapshot, 2 for latent

labevents_stats = pd.read_parquet('../Datasets/labevents_stats.parquet')
microbiology_stats = pd.read_parquet('../Datasets/microbiology_stats.parquet')
prescriptions_stats = pd.read_parquet('../Datasets/prescriptions_stats.parquet')
radiology_stats = pd.read_parquet('../Datasets/radiology_stats.parquet')

data_config = {
    'num_events': 3,
    'num_subevents':{
        'n_lab': labevents_stats['itemid'].nunique(),
        'n_micro': microbiology_stats['itemid'].nunique(),
        'n_pres': prescriptions_stats['itemid'].nunique(),
        'n_radio': radiology_stats['itemid'].nunique(),
    }
}
