from fmif.model_utils import ProteinMPNNFMIF, loss_smoothed, loss_nll
from fmif.fm_utils import fm_model_step
import torch 

class ProteinSDPO(ProteinMPNNFMIF):
    def __init__(self, ref_model=None, beta=1.0, eval=False, generator='rkl', node_features=128, edge_features=128,
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=22, k_neighbors=32, augment_eps=0.1, dropout=0.1, cfg=False):
        super(ProteinSDPO, self).__init__(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            vocab=vocab,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=dropout,
            cfg=cfg
        )
        
        self.beta = beta
        self.generator = generator
        if ref_model != None:
            self.set_ref_model(ref_model)
        
    def set_ref_model(self, ref_model):
        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, cls=None):
        log_probs = super(ProteinSDPO, self).forward(X, S, mask, chain_M, residue_idx, chain_encoding_all, cls)
        return log_probs

def sdpo_loss(fpo_model, S, noisy_batch, mask_for_loss, rewards):
    log_probs_ft = fm_model_step(fpo_model, noisy_batch)
    loss_ft, _, _ = loss_nll(S, log_probs_ft, mask_for_loss) # B x L        
    model_log_prob = loss_ft.sum(dim=-1) # B x 1
    
    log_probs_ref = fm_model_step(fpo_model.ref_model, noisy_batch)
    loss_ref, _, _ = loss_nll(S, log_probs_ref, mask_for_loss) # B x L        
    ref_log_prob = loss_ref.sum(dim=-1) # B x 1
    
    g_theta = -fpo_model.beta * (model_log_prob - ref_log_prob)
        
    if fpo_model.generator == 'rkl':
        lsm = torch.log_softmax(rewards, dim=0) - torch.log_softmax(g_theta, dim=0)
        loss = (torch.nn.functional.softmax(rewards, dim=0) * lsm).sum()
    
    return loss
    