"""ESM-2 residues + MolUNet atoms + online DP1 joint-graph fusion."""
import copy

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.utils import to_dense_batch

from .dynamic_pocket import DynamicPocketSelectorLite
from .dmff_layers import GINConvNet, LinkAttention
from .molunet import MolUnetEncoder


def masked_mean(values, mask):
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


class DynaMolDTA(nn.Module):
    def __init__(self, config, deg, smiles_vocab_size=46):
        super().__init__()
        self.config = copy.deepcopy(config)
        dim = config['hidden_dim']
        dropout = config['dropout']
        self.mode = config['pocket_mode']
        if self.mode not in ('full_online', 'static_sparse', 'dynamic_pocket'):
            raise ValueError('pocket_mode must be full_online, static_sparse or dynamic_pocket')
        if dim % 2 or not 0 <= config['gate_floor'] < 1:
            raise ValueError('hidden_dim must be even; gate_floor must be in [0, 1)')
        self.protein_encoder = nn.Sequential(
            nn.Linear(config['esm_dim'], dim), nn.LayerNorm(dim), nn.Dropout(dropout))
        mol_config = copy.deepcopy(config['molunet'])
        mol_config['deg'] = deg
        self.molunet = MolUnetEncoder(mol_config)
        mol_dim = mol_config['model']['hidden_dim']
        self.mol_query = nn.Linear(dim, mol_config['FP']['hidden_dims'][-1])
        self.atom_projection = nn.Sequential(
            nn.Linear(mol_dim * mol_config['model']['num_pool_layer'], dim), nn.LayerNorm(dim))
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, 256, padding_idx=0)
        self.smiles_input = nn.Linear(256, dim // 2)
        self.smiles_encoder = nn.LSTM(dim // 2, dim // 2, num_layers=2,
                                      batch_first=True, bidirectional=True, dropout=dropout)
        self.smiles_norm = nn.LayerNorm(dim)
        self.selector = DynamicPocketSelectorLite(
            residue_dim=dim, atom_dim=dim, hidden_dim=128, n_heads=4,
            target_ratio=config['pocket_ratio'], temperature=config['pocket_temperature'])
        self.smiles_attention = LinkAttention(dim, 8)
        self.protein_attention = LinkAttention(dim, 8)
        self.joint_attention = LinkAttention(dim, 8)
        self.sequence_projection = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), nn.ReLU(), nn.Dropout(dropout))
        self.joint_gin = GINConvNet(num_features_xd=dim + 1, n_output=dim * 2,
                                    dropout=dropout)
        self.predictor = nn.Sequential(
            nn.Linear(dim * 4, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, 1))

    def forward(self, batch, return_auxiliary=False):
        residue_mask = batch['protein_mask']
        residues = self.protein_encoder(batch['protein_esm'])
        residues = residues * residue_mask.unsqueeze(-1)
        query = self.mol_query(masked_mean(residues, residue_mask))
        scales, _, _ = self.molunet(batch['drug_graph'], query=query)
        atoms_flat = self.atom_projection(torch.cat(scales, dim=-1))
        atoms, atom_mask = to_dense_batch(atoms_flat, batch['drug_graph'].batch)

        if self.mode == 'full_online':
            pocket = {'pocket_scores': residue_mask.to(residues.dtype),
                      'pocket_mask': residue_mask}
        else:
            pocket = self.selector(residues, atoms, residue_mask, atom_mask,
                                   ligand_conditioned=self.mode == 'dynamic_pocket',
                                   supervised=self.config['pocket_loss_weight'] > 0)
        gates = self.config['gate_floor'] + (1 - self.config['gate_floor']) * pocket['pocket_scores']
        gated_residues = residues * gates.unsqueeze(-1)

        # The two PyG batches have different topologies and must never share edge_index.
        node_blocks = []
        for i in range(residues.shape[0]):
            protein_nodes = gated_residues[i, residue_mask[i]]
            drug_nodes = atoms[i, atom_mask[i]]
            hub = (protein_nodes.mean(0, keepdim=True) + drug_nodes.mean(0, keepdim=True)) / 2
            features = torch.cat([protein_nodes, drug_nodes, hub])
            node_type = features.new_zeros((len(features), 1))
            node_type[len(protein_nodes):-1] = 1
            node_type[-1] = 2
            node_blocks.append(torch.cat([features, node_type], dim=-1))
        graph = batch['joint_graph']
        x = torch.cat(node_blocks)
        if x.shape[0] != graph.num_nodes:
            raise ValueError('Joint graph nodes are not aligned with residue/atom features')
        graph_output = self.joint_gin(x, graph.edge_index, graph.batch)

        tokens = batch['smiles_tokens']
        smiles_mask = batch['smiles_mask']
        embedded = self.smiles_input(self.smiles_embedding(tokens))
        packed = pack_padded_sequence(embedded, smiles_mask.sum(1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed, _ = self.smiles_encoder(packed)
        smiles, _ = pad_packed_sequence(packed, batch_first=True, total_length=tokens.shape[1])
        smiles = self.smiles_norm(smiles)
        sm_mask = smiles_mask[:, None, :].expand(-1, 8, -1)
        pr_mask = residue_mask[:, None, :].expand(-1, 8, -1)
        sm_out, _ = self.smiles_attention(smiles, sm_mask)
        pr_out, _ = self.protein_attention(residues, pr_mask)
        pair_out, _ = self.joint_attention(torch.cat([smiles, residues], dim=1),
                                          torch.cat([sm_mask, pr_mask], dim=-1))
        sequence_output = self.sequence_projection(torch.cat([sm_out, pr_out, pair_out], dim=-1))
        prediction = self.predictor(torch.cat([graph_output, sequence_output], dim=-1)).squeeze(-1)
        return (prediction, pocket) if return_auxiliary else prediction


def pocket_loss(batch, auxiliary):
    logits = auxiliary.get('decision_logits')
    valid = batch['pocket_label_valid']
    if logits is None or not valid.any():
        return batch['y'].new_zeros(())
    mask = batch['protein_mask'].to(logits.dtype)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, batch['pocket_label'], reduction='none')
    per_sample = (loss * mask).sum(1) / mask.sum(1).clamp_min(1)
    return per_sample[valid].mean()
