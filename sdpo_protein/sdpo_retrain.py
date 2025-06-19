import argparse
import random
import string
import datetime
import pickle
from protein_oracle.utils import str2bool
from fmif.fm_utils import Interpolant, fm_model_step
from eval import eval_model

runid = ''.join(random.choice(string.ascii_letters) for i in range(10))+ '_' + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))


def main(args):
    import os
    import warnings
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import os.path
    from concurrent.futures import ProcessPoolExecutor    
    from protein_oracle.utils import set_seed
    from protein_oracle.data_utils import ProteinStructureDataset, ProteinDPODataset, featurize
    from protein_oracle.model_utils import ProteinMPNNOracle
    from fmif.model_utils import ProteinMPNNFMIF, loss_smoothed, loss_nll
    from sdpo_protein import ProteinSDPO, sdpo_loss
    from tqdm import tqdm
    import wandb
    warnings.filterwarnings("ignore", category=UserWarning)
     
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    path_for_outputs = os.path.join(args.base_path, 'protein_rewardbp')
    
    if args.wandb:
        run = wandb.init(
            project="Protein FPO", 
            id=os.environ.get("WANDB_RUN_ID"), 
            resume="allow",
            config={
                'lr': args.lr,
                'K': args.K,
                'beta': args.beta,
                'num_epochs': args.num_epochs
            }
        )   
    
    print(f'lr: {args.lr},  K: {args.K}, beta: {args.beta}, num_epochs: {args.num_epochs}')

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
    # dpo_train_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_train_dict_curated.pkl'), 'rb'))
    dpo_train_dict = torch.load(os.path.join(dpo_dict_path, args.data_path))
    dpo_test_dict = pickle.load(open(os.path.join(dpo_dict_path, 'dpo_test_dict_wt.pkl'), 'rb'))
    
    dpo_train_dict_complete = dpo_train_dict

    dpo_train_dataset = ProteinDPODataset(dpo_train_dict_complete, pdb_idx_dict, pdb_structures)
    loader_train = DataLoader(dpo_train_dataset, batch_size=args.K, shuffle=True)

    dpo_test_dataset = ProteinDPODataset(dpo_test_dict, pdb_idx_dict, pdb_structures)
    loader_test = DataLoader(dpo_test_dataset, batch_size=1, shuffle=False)
    
    # print(dpo_train_dataset.__getitem__(0))

    if args.initialize_with_pretrain:
        ref_model = ProteinMPNNFMIF(node_features=args.hidden_dim,
                            edge_features=args.hidden_dim,
                            hidden_dim=args.hidden_dim,
                            num_encoder_layers=args.num_encoder_layers,
                            num_decoder_layers=args.num_encoder_layers,
                            k_neighbors=args.num_neighbors,
                            dropout=args.dropout,
                            augment_eps=args.backbone_noise
                            )
        ref_model.to(device)
        if args.use_pretrained_ref:
            ref_model.load_state_dict(torch.load(os.path.join(args.base_path, 'pmpnn/outputs/pretrained_if_model.pt'))['model_state_dict'])
        else:
            ref_model.finetune_init()
            ref_model.load_state_dict(torch.load(os.path.join(args.base_path, args.model_path)))
        
        fmif_model = ProteinSDPO(node_features=args.hidden_dim,
                                beta=args.beta,
                            edge_features=args.hidden_dim,
                            hidden_dim=args.hidden_dim,
                            num_encoder_layers=args.num_encoder_layers,
                            num_decoder_layers=args.num_encoder_layers,
                            k_neighbors=args.num_neighbors,
                            dropout=args.dropout,
                            augment_eps=args.backbone_noise
                            )
        fmif_model.to(device)
        fmif_model.finetune_init()
        fmif_model.load_state_dict(torch.load(os.path.join(args.base_path, args.model_path)))
        fmif_model.set_ref_model(ref_model)
        fmif_model.train()
        
        print(f'Checkpoint loaded from {os.path.join(args.base_path, args.data_path)}')

    optimizer = torch.optim.AdamW(fmif_model.parameters(), lr=args.lr, weight_decay=args.wd)

    noise_interpolant = Interpolant(args)
    noise_interpolant.set_device(device)
    
    print(len(loader_train))
    
    reward_model_eval = ProteinMPNNOracle(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        )
    reward_model_eval.to(device)
    reward_model_eval.load_state_dict(torch.load(os.path.join(args.base_path, 'protein_oracle/outputs/reward_oracle_eval.pt'))['model_state_dict'])
    reward_model_eval.finetune_init()
    reward_model_eval.eval()

    pbar = tqdm(range(args.num_epochs))
    for e in pbar:
        total_loss = 0.
        pbar2 = tqdm(enumerate(loader_train))
        for _, batch in pbar2:
            if args.debug and _ > 20:
                break
            X, S, mask, chain_M, residue_idx, chain_encoding_all, S_wt = featurize(batch, device)
            mask_for_loss = mask * chain_M
            dg_ml = batch['dG_ML'].to(dtype=torch.float32, device=device)
            dg_ml_wt = batch['dG_ML_wt'].to(dtype=torch.float32, device=device)
            ddg_ml = dg_ml - dg_ml_wt # reward B x 1
            noisy_batch = noise_interpolant.corrupt_batch((X, S, mask, chain_M, residue_idx, chain_encoding_all))
            
            loss = sdpo_loss(fmif_model, S, noisy_batch, mask_for_loss, ddg_ml)
            
            if args.zero_grad:
                optimizer.zero_grad()
            
            loss.backward()
            if args.gradient_clip_val > 0.:
                torch.nn.utils.clip_grad_norm_(
                    fmif_model.parameters(), max_norm=args.gradient_clip_val)
            optimizer.step()
            
            total_loss += loss.item()
            pbar2.set_description(f'Batch: {_}. Train loss: {loss.item()}')
        
        avg_loss = total_loss / len(loader_train)
        pbar.set_description(
            (
                f'Epoch: {e}. Avg. Train loss: {avg_loss}'
            )
        )
        
        if (e + 1) % args.eval_every == 0:
            fmif_model.eval()
            rewards_eval = eval_model(fmif_model, reward_model_eval, loader_test, device, noise_interpolant, rmsd=False)
            fmif_model.train()
    
    rewards_eval = eval_model(fmif_model, reward_model_eval, loader_test, device, noise_interpolant, rmsd=False)
    if wandb.run is not None:
        wandb.log({'mean_reward': rewards_eval.mean()})
        wandb.log({'median_reward': np.median(rewards_eval)})
        wandb.log({'Positive reward proportion': np.mean(rewards_eval>0)})
    
    fmif_model.ref_model = None
    weights_path = os.path.join(path_for_outputs, args.save_path)
    torch.save(fmif_model.state_dict(), weights_path)
    
    print(f'Done fine tuning! Model weights saved to {weights_path}. Max reward value: {rewards_eval.max()}')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--base_path", type=str, default="data_and_model/", help="base path for data and model") 
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=2, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--K", type=int, default=128, help="number of sequences for one batch")   # TODO
    argparser.add_argument("--batch_size", type=int, default=128, help="number of sequences for one batch")   # TODO
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
    argparser.add_argument("--min_t", type=float, default=1e-2)
    argparser.add_argument("--schedule", type=str, default='linear') # other schedule is not implemented
    argparser.add_argument("--schedule_exp_rate", type=float, default=-3)
    argparser.add_argument("--temp", type=float, default=0.1)
    argparser.add_argument("--noise", type=float, default=1.0)
    argparser.add_argument("--interpolant_type", type=str, default='masking')
    argparser.add_argument("--num_timesteps", type=int, default=50) # 500
    argparser.add_argument("--lr", type=float, default=3e-5, help="optimization learning rate")
    argparser.add_argument("--wd", type=float, default=0, help="optimization weight decay")
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--save_path", type=str, default='protein_fpo_r2.pt', help="filename for finetuned model weights")
    argparser.add_argument("--beta", type=float, default=1.0, help="beta regularization parameter")
    argparser.add_argument("--gradient_clip_val", type=float, default=1.0, help="gradient clipping during finetuning")
    argparser.add_argument("--zero_grad", type=bool, default=True, help="toggle zero grad")
    argparser.add_argument("--data_path", type=str, default='0.075beta_relabel.pt', help="path to dataset")
    argparser.add_argument("--model_path", type=str, default='protein_rewardbp/0.075beta.pt', help="path to model")
    argparser.add_argument("--eval_every", type=int, default=10, help='eval every _ epochs')
    argparser.add_argument("--use_pretrained_ref", type=bool, default=False, help='toggle use original pre-trained model as reference model')
 
    args = argparser.parse_args()
    print(args)
    main(args)
