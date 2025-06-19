import argparse
import random
import string
import datetime
import pickle
from protein_oracle.utils import str2bool
from fmif.fm_utils import Interpolant, fm_model_step
from eval import eval_model
import copy


def main(args):
    import os
    import warnings
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, random_split
    import torch.nn.functional as F
    import os.path
    from concurrent.futures import ProcessPoolExecutor    
    from protein_oracle.utils import set_seed
    from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize, batch_ints_to_strings
    from protein_oracle.model_utils import ProteinMPNNOracle
    from fmif.model_utils import ProteinMPNNFMIF, loss_smoothed, loss_nll
    from sdpo_protein import ProteinSDPO, sdpo_loss
    from tqdm import tqdm
    import wandb
    warnings.filterwarnings("ignore", category=UserWarning)
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    path_for_outputs = os.path.join(args.base_path, 'proteindpo_data/processed_data')
    
    assert torch.cuda.is_available(), "CUDA is not available"
    set_seed(args.seed, use_cuda=True)

    pdb_path = os.path.join(args.base_path, 'proteindpo_data/AlphaFold_model_PDBs')
    max_len = 75  # Define the maximum length of proteins
    dataset = ProteinStructureDataset(pdb_path, max_len) # max_len set to 75 (sequences range from 31 to 74)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    # make a dict of pdb filename: index
    for batch in loader:
        pdb_structures = batch[0]
        pdb_filenames = batch[1]
        pdb_idx_dict = {pdb_filenames[i]: i for i in range(len(pdb_filenames))}
        break

    dpo_dict_path = os.path.join(args.base_path, 'proteindpo_data/processed_data')
    dpo_train_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict_curated.pkl'), 'rb'))
    dpo_valid_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_valid_dict.pkl'), 'rb'))
    dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
    
    if args.include_all:
        dpo_train_dict_complete = {**dpo_train_dict, **dpo_valid_dict, **dpo_test_dict}
    else:
        dpo_train_dict_complete = dpo_train_dict

    dpo_train_dataset = ProteinDPODataset(dpo_train_dict_complete, pdb_idx_dict, pdb_structures)
        
    random_indices = torch.randperm(len(dpo_train_dataset))[:args.num_data].tolist()
    
    # Select protein names corresponding to the random indices
    subset_keys = [dpo_train_dataset.protein_list[i] for i in random_indices]
    
    new_dpo_train_dict = {key: dpo_train_dataset.dpo_train_dict[key] for key in subset_keys}
    
    # new_structures_dataset, _ = random_split(dpo_train_dataset, [args.num_data, len(dpo_train_dataset) - args.num_data])
    new_structures_dataset = ProteinDPODataset(new_dpo_train_dict, dpo_train_dataset.pdb_idx_dict, dpo_train_dataset.pdb_structure)
    structures_loader = DataLoader(new_structures_dataset, batch_size=args.batch_size, shuffle=True)

    # dpo_test_dataset = ProteinDPODataset(dpo_test_dict, pdb_idx_dict, pdb_structures)
    # loader_test = DataLoader(dpo_test_dataset, batch_size=1, shuffle=False)
    
    if args.initialize_with_pretrain:
        # fmif_model = ProteinFPO(node_features=args.hidden_dim,
        #                     edge_features=args.hidden_dim,
        #                     hidden_dim=args.hidden_dim,
        #                     num_encoder_layers=args.num_encoder_layers,
        #                     num_decoder_layers=args.num_encoder_layers,
        #                     k_neighbors=args.num_neighbors,
        #                     dropout=args.dropout,
        #                     augment_eps=args.backbone_noise
        #                     )
        fmif_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
        fmif_model.to(device)
        # fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, 'pmpnn/outputs/pretrained_if_model.pt'))['model_state_dict'])
        fmif_model.finetune_init()
        # fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, args.model_path))['model_state_dict'])
        fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, args.model_path)))
        fmif_model.eval()

    noise_interpolant = Interpolant(args)
    noise_interpolant.set_device(device)
    
    reward_model_train = ProteinMPNNOracle(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
    reward_model_train.to(device)
    reward_model_train.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_oracle/outputs/reward_oracle_ft.pt'))['model_state_dict'])
    reward_model_train.finetune_init()
    reward_model_train.eval()
        
    pbar = tqdm(structures_loader)
    
    new_protein_dict = copy.deepcopy(new_structures_dataset.dpo_train_dict)
    
    for batch in pbar:

        X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
                
        S_sp, _, _ = noise_interpolant.sample(fmif_model, X, mask, chain_M, residue_idx, chain_encoding_all)
        ddg_pred = reward_model_train(X, S_sp, mask, chain_M, residue_idx, chain_encoding_all)
        seqs = batch_ints_to_strings(S_sp)
        
        name = batch['protein_name']
        new_protein_dict[name[0]][0] = seqs[0]
        new_protein_dict[name[0]][3] = ddg_pred.item()
        new_protein_dict[name[0]][5] = 0.
        
        pbar.set_description(f'{ddg_pred.item()}')
        
    # new_dataset = ProteinDPODataset(new_protein_dict, dpo_train_dataset.pdb_idx_dict, dpo_train_dataset.pdb_structure)
    torch.save(new_protein_dict, os.path.join(args.base_path, args.save_path))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--base_path", type=str, default="data_and_model/", help="base path for data and model") 
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=2, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--batch_size", type=int, default=1, help="number of sequences for one batch")   # TODO
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=30, help="number of neighbors for the sparse graph")   # 48
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")   # TODO
    argparser.add_argument("--backbone_noise", type=float, default=0.1, help="amount of noise added to backbone during training")   # TODO
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=str2bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--mixed_precision", type=str2bool, default=True, help="train with mixed precision")
    argparser.add_argument("--initialize_with_pretrain", type=str2bool, default=True, help="initialize with FMIF weights")
    argparser.add_argument("--include_all", type=str2bool, default=False, help="include valid and test into training, for evaluation oracle")
    argparser.add_argument("--wandb", type=bool, default=False, help="toggle wandb logging")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--min_t", type=float, default=1e-2)
    argparser.add_argument("--schedule", type=str, default='linear') # other schedule is not implemented
    argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
    argparser.add_argument("--temp", type=float, default=0.1)
    argparser.add_argument("--noise", type=float, default=1.0)
    argparser.add_argument("--interpolant_type", type=str, default='masking')
    argparser.add_argument("--num_timesteps", type=int, default=50) # 500
    argparser.add_argument("--model_path", type=str, default='pmpnn/outputs/pretrained_if_model.pt')
    argparser.add_argument("--num_data", type=int, default=100000)
    argparser.add_argument("--save_path", type=str, default='proteindpo_data/processed_data/sdpo_r2_dict.pt')
 
    args = argparser.parse_args()    
    print(args)
    main(args)
