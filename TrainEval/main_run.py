import time
import logging
import numpy as np
import torch
from torch import nn

from Datasets.observe_intervene_dataloader import get_dataloader
from Loss.loss_function import ObserveInterveneLoss
from Loss.positive_weights import get_positive_weights
from Networks.EpisodeEmbedder import EpisodeEmbedderModule
from Networks.StateModel import ClinicalStateMachineGRU
from Networks.StateTransformer import StateTransformer
from TrainEval.config import model_type
from TrainEval.train_val import train_one_epoch, validate
from config import config, data_config, episodic_data_path, embedding_path, cxr_path, train_path, val_path, \
    log_config_filepath

torch.manual_seed(28)
np.random.seed(28)
torch.cuda.manual_seed(28)
torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(filename=log_config_filepath, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')
    logger.info(f'Model Type training : {model_type}')

    train_dl = get_dataloader(episodic_data_path, train_path, embedding_path, cxr_path, size=-1)
    val_dl = get_dataloader(episodic_data_path, val_path, embedding_path, cxr_path, size=-1)
    logger.info(f'Train size: {len(train_dl)}, val size: {len(val_dl)}')

    if model_type == 'transformers':
        state_model = StateTransformer(
            num_events=data_config['num_events'],
            num_subevents=data_config['num_subevents'],
            input_dim=config['hidden_dim'],
            hidden_dim=config['hidden_dim']
        ).to(device)
    else:
        state_model = ClinicalStateMachineGRU(
            num_events=data_config['num_events'],
            num_subevents=data_config['num_subevents'],
            embed_dim=config['embed_dim'],
            hidden_dim=config['hidden_dim'],
        ).to(device)

    episode_embedding_model = EpisodeEmbedderModule(
        device=device,
        num_prescriptions = data_config['num_subevents']['n_pres'],
        num_labs = data_config['num_subevents']['n_lab'],
        num_micro_labs = data_config['num_subevents']['n_micro'],
        num_exam_codes = data_config['num_subevents']['n_radio'],
        lab_embed_dim = config['lab_embed_dim'],
        output_dim = config['embed_dim']
    ).to(device)

    get_positive_weights(episodic_data_path)
    loss_fn = ObserveInterveneLoss(device).to(device)
    optimizer = torch.optim.Adam((list(state_model.parameters()) + list(episode_embedding_model.parameters())), lr=config['lr'])

    init_state = nn.Parameter(torch.empty(config['batch_size'], config['hidden_dim'])).to(device)

    if config['resume']:
        state_model.load_state_dict(torch.load(config['load_state_model_path']))
        episode_embedding_model.load_state_dict(torch.load(config['load_episode_embedding_model_path']))
        init_state = torch.load(config['load_init_state_path'])
    else:
        nn.init.xavier_uniform_(init_state)

    for epoch in range(config['init_epoch'], config['epochs']):
        start_time = time.time()
        train_loss, train_metrics = train_one_epoch(device, epoch, state_model, episode_embedding_model, init_state, train_dl, optimizer, loss_fn, config['accumulation_steps'])
        end_time = time.time()
        logger.info(f'Train loss: {train_loss}, time: {round(end_time - start_time, 4)}')
        logger.info(f'Train metrics: {train_metrics}')

        torch.save(state_model.state_dict(), f'{config['model_save_dir']}/state_model_{epoch}.pth')
        torch.save(episode_embedding_model.state_dict(),f'{config['model_save_dir']}/episode_embedding_model_{epoch}.pth')
        torch.save(init_state, f'{config['model_save_dir']}/init_state_{epoch}.pth')

        if epoch % config['val_every'] == 0:
            val_metrics = validate(state_model, episode_embedding_model, init_state, val_dl)
            logger.info(f'Validation Metrics at epoch {epoch}: {val_metrics}')


if __name__ == '__main__':
    main()