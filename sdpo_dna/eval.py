import os
import diffusion_gosai_update
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from argparse import ArgumentParser
import dataloader_gosai
import numpy as np
import pandas as pd
import oracle
from scipy.stats import pearsonr
import torch
from tqdm import tqdm
from utils import set_seed
from grelu.interpret.motifs import scan_sequences
from sdpo_gosai import DiffusionSDPO

def compare_kmer(kmer1, kmer2, n_sp1, n_sp2):
    kmer_set = set(kmer1.keys()) | set(kmer2.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in kmer1:
            counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
        if kmer in kmer2:
            counts[i][0] = kmer2[kmer]
    return pearsonr(counts[:, 0], counts[:, 1])[0]


def get_model(base_path, ckpt_path, config_path='configs_gosai', config_name='config_gosai.yaml'):
    GlobalHydra.instance().clear()

    # Initialize Hydra and compose the configuration
    initialize(config_path=config_path, job_name="load_model")
    cfg = compose(config_name=config_name)
    cfg.eval.checkpoint_path = os.path.join(base_path, ckpt_path)

    model = diffusion_gosai_update.Diffusion(cfg, eval=False).cuda() # Finetuned model
    model.load_state_dict(torch.load(cfg.eval.checkpoint_path))
    model.eval()
    return model

def eval_model(model, old_model, num_sample_batches, num_samples_per_batch, verbose=True, jaspar=False):
    all_detokenized_samples = []
    all_raw_samples = []
    for _ in tqdm(range(num_sample_batches)):
        samples = model._sample(eval_sp_size=num_samples_per_batch)
        all_raw_samples.append(samples)
        detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy())
        all_detokenized_samples.extend(detokenized_samples)
    all_raw_samples = torch.concat(all_raw_samples)
    
    highexp_kmers_99, n_highexp_kmers_99, highexp_kmers_999, n_highexp_kmers_999, highexp_set_sp_clss_999, highexp_preds_999, highexp_seqs_999 = oracle.cal_highexp_kmers(return_clss=True)

    
    if jaspar:
        motif_count = scan_sequences(all_detokenized_samples, 'jaspar')
        motif_count_sum = motif_count['motif'].value_counts()
        motif_count_top = scan_sequences(highexp_seqs_999, 'jaspar')
        motif_count_top_sum = motif_count_top['motif'].value_counts()
        
        motifs_summary = pd.concat([motif_count_top_sum, motif_count_sum], axis=1)
        motifs_summary.columns = ['top_data', 'finetuned']
        if verbose:
            print(motifs_summary.corr(method='spearman'))
        jaspar_corrs = motifs_summary.corr(method='spearman')
    else:
        jaspar_corrs = None
        
    model_logl = old_model.get_likelihood(all_raw_samples, num_steps=128, n_samples=1)
    
    if verbose:
        print(f"Mpdel's sample avg. log-likelihood: {model_logl.mean()}, Median log-likelihood: {np.median(model_logl.detach().cpu())}")
    
    generated_preds = oracle.cal_gosai_pred_new(all_detokenized_samples, mode='eval')
    
    if verbose:
        print(f'Model avg. generated sample predicted HepG2: {generated_preds[:, 0].mean()}, median predicted HepG2: {np.median(generated_preds[:, 0])}')
    
    generated_preds_atac = oracle.cal_atac_pred_new(all_detokenized_samples)
    
    generated_atac_acc = (generated_preds_atac[:,1] > 0.5).sum() / (num_sample_batches * num_samples_per_batch)
    
    if verbose:
        print(f'Model ATAC accuracy: {generated_atac_acc}')
    
    generated_kmer = oracle.count_kmers(all_detokenized_samples)
    
    generated_p_coef = compare_kmer(highexp_kmers_999, generated_kmer, n_highexp_kmers_999, len(all_detokenized_samples))
    
    if verbose:
        print(f'Model generated 3-mer Pearson correlation: {generated_p_coef}')
    
    if jaspar:
        return all_detokenized_samples, model_logl, generated_preds, generated_atac_acc, generated_p_coef, jaspar_corrs
    return all_detokenized_samples, model_logl, generated_preds, generated_atac_acc, generated_p_coef

def eval_model_from_paths(base_path, ckpt_path, num_sample_batches, num_samples_per_batch, 
               config_path='configs_gosai', config_name='config_sdpo_gosai.yaml', ref_path='mdlm/outputs_gosai/pretrained.ckpt',
               jaspar=False):

    GlobalHydra.instance().clear()
    
    initialize(config_path=config_path, job_name="load_model")
    config = compose(config_name=config_name)
    
    ref_path = os.path.join(base_path, ref_path)
    
    ref_model = DiffusionSDPO.load_from_checkpoint(ref_path, config=config, beta=1.0, generator='rkl').to('cuda')
    ref_model.eval()
    
    model_path = os.path.join(base_path, ckpt_path)
    
    model = DiffusionSDPO.load_from_checkpoint(ref_path, config=config, beta=1.0)
    # model.set_ref_model(ref_model)
    model.ref_model = None
    model.load_state_dict(torch.load(model_path), strict=False)
    
    return eval_model(model, ref_model, num_sample_batches, num_samples_per_batch, jaspar=jaspar)
    
    
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base_path', type=str, default='data_and_model/')
    parser.add_argument('--model_path', type=str, default='mdlm/reward_bp_results_final/finetuned.ckpt')
    parser.add_argument('--ref_model_path', type=str, default='mdlm/outputs_gosai/pretrained.ckpt')
    parser.add_argument('--num_samples_per_batch', type=int, default=64)
    parser.add_argument('--num_sample_batches', type=int, default=10)
    parser.add_argument('--jaspar', type=bool, default=False)
    args = parser.parse_args()
    
    set_seed(args.seed, use_cuda=True)
    
    results = eval_model_from_paths(args.base_path, args.model_path, args.num_sample_batches, args.num_samples_per_batch, ref_path=args.ref_model_path, jaspar=args.jaspar)