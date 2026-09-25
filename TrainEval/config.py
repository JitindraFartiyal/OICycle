import pandas as pd

episodic_data_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed\episodic_master_table.parquet"
embedding_path = r"D:\data\Project Medical AI\datasets_v2.0\preprocessed"
cxr_path = r"D:\data\Project Medical AI\datasets_v2.0\cxr"

train_path = r"D:\data\Project Medical AI\datasets_v2.0\train_subjects.csv"
val_path = r"D:\data\Project Medical AI\datasets_v2.0\val_subjects.csv"
test_path = r"D:\data\Project Medical AI\datasets_v2.0\test_subjects.csv"
log_config_filepath = r'../logs/oci_transformers.log'

model_type = 'transformers' #POMDP, single_snapshot, transformers, latent

config = {
    'init_epoch': 0,
    'epochs':10,
    'lr': 1e-3,
    'batch_size': 1,
    'hidden_dim': 512,
    'lab_embed_dim': 128,
    'embed_dim': 512,
    'threshold': 0.5,
    'accumulation_steps': 128,
    'print_every': 500,
    'val_every':1,
    'lambdas': {
        'loss_t1': 1., 'loss_t2': 1., 'loss_t3': 1.,
        'loss_type': 1., 'loss_pres': 1., 'loss_lab': 1., 'loss_micro': 1., 'loss_radio': 1.
    },
    'model_save_dir': r"D:\data\Project Medical AI\code_v2.0\ckpts_transformers",
    'resume': False,
    'load_state_model_path': r'',
    'load_episode_embedding_model_path': r'',
    'load_init_state_path': r''
}

labevents_stats = pd.read_parquet('../Datasets/labevents_stats.parquet')
microbiology_stats = pd.read_parquet('../Datasets/microbiology_stats.parquet')
prescriptions_stats = pd.read_parquet('../Datasets/prescriptions_stats.parquet')
radiology_stats = pd.read_parquet('../Datasets/radiology_stats.parquet')

intervention_type_weights_path = r'../Loss/intervention_type_weights.npz'
intervention_value_weights_path = r'../Loss/intervention_value_weights.npz'

data_config = {
    'num_events': 3,
    'num_subevents':{
        'n_lab': labevents_stats['itemid'].nunique(),
        'n_micro': microbiology_stats['itemid'].nunique(),
        'n_pres': prescriptions_stats['itemid'].nunique(),
        'n_radio': radiology_stats['itemid'].nunique(),
    }
}


