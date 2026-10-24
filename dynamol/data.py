"""Aligned drug atoms, protein windows and frozen ESM-2 residue features."""
import hashlib
import inspect
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from rdkit import Chem

from .build_vocab import WordVocab
from .molunet.utils import x_map, e_map


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path):
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)


class VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'WordVocab' and module in ('__main__', 'build_vocab'):
            return WordVocab
        return super().find_class(module, name)


def load_vocab(path):
    with open(path, 'rb') as handle:
        vocab = VocabUnpickler(handle).load()
    if vocab.pad_index != 0:
        raise ValueError('The model requires padding token index 0')
    return vocab


def tokenize(smiles, vocab):
    tokens, i = [vocab.sos_index], 0
    while i < len(smiles):
        width = 2 if i + 1 < len(smiles) and smiles[i:i + 2] in vocab.stoi else 1
        tokens.append(vocab.stoi.get(smiles[i:i + width], vocab.unk_index))
        i += width
    return torch.tensor(tokens + [vocab.eos_index], dtype=torch.long)


def molecule_graph(smiles):
    """RDKit atom order is used by both MolUNet and the joint graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Invalid or empty SMILES: {smiles}')
    atoms = []
    for atom in mol.GetAtoms():
        values = [atom.GetAtomicNum(), str(atom.GetChiralTag()), atom.GetTotalDegree(),
                  atom.GetFormalCharge(), atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(),
                  str(atom.GetHybridization()), atom.GetIsAromatic(), atom.IsInRing()]
        atoms.append([choices.index(value) for choices, value in zip(x_map.values(), values)])
    edges, attrs = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        values = [str(bond.GetBondType()), str(bond.GetStereo()), bond.GetIsConjugated()]
        attr = [choices.index(value) for choices, value in zip(e_map.values(), values)]
        edges.extend([[a, b], [b, a]])
        attrs.extend([attr, attr])
    if not edges:
        raise ValueError(f'MolUNet currently requires at least one molecular bond: {smiles}')
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attrs, dtype=torch.long)
    order = (edge_index[0] * len(atoms) + edge_index[1]).argsort()
    return Data(x=torch.tensor(atoms, dtype=torch.long), edge_index=edge_index[:, order],
                edge_attr=edge_attr[order], num_nodes=len(atoms))


def joint_graph(drug, contact_edges, residue_count):
    atom_count = drug.num_nodes
    hub = residue_count + atom_count
    nodes = torch.arange(hub + 1)
    edges = torch.cat([contact_edges, drug.edge_index + residue_count,
                       torch.stack([nodes, nodes]),
                       torch.stack([torch.arange(hub), torch.full((hub,), hub)]),
                       torch.stack([torch.full((hub,), hub), torch.arange(hub)])], dim=1)
    return Data(edge_index=torch.unique(edges, dim=1), num_nodes=hub + 1)


class AffinityDataset(Dataset):
    def __init__(self, csv, esm_dir, contact_dir, vocab_path, esm_dim=640,
                 contact_threshold=0.8, pocket_labels=None):
        self.csv = Path(csv).resolve()
        self.frame = pd.read_csv(self.csv)
        required = {'compound_iso_smiles', 'target_sequence', 'target_key', 'affinity',
                    'target_sequence_start', 'target_sequence_end'}
        if not required.issubset(self.frame.columns):
            raise ValueError(f'Missing CSV columns: {sorted(required - set(self.frame.columns))}')
        if len(self.frame) == 0 or self.frame[list(required)].isna().any().any():
            raise ValueError('CSV must contain nonempty rows with all required values')
        if not np.isfinite(self.frame['affinity'].astype(float)).all():
            raise ValueError('Affinity labels must be finite')
        self.vocab = load_vocab(vocab_path)
        self.manifest = {'csv': file_hash(csv), 'vocab': file_hash(vocab_path),
                         'esm_dim': esm_dim, 'contact_threshold': contact_threshold,
                         'esm': {}, 'contacts': {}}
        self.drugs = {}
        self.tokens = {}
        for sm in self.frame['compound_iso_smiles'].unique():
            self.drugs[sm] = molecule_graph(sm)
            self.tokens[sm] = tokenize(sm, self.vocab)
        self.proteins, self.joints = {}, {}
        labels = {}
        if pocket_labels:
            with np.load(pocket_labels, allow_pickle=False) as archive:
                labels = {key: archive[key] for key in archive.files}
            self.manifest['pocket_labels'] = file_hash(pocket_labels)
        self.labels = {}
        seen_ids = {}
        for _, row in self.frame.iterrows():
            seq = row['target_sequence']
            target = str(row['target_key'])
            uniprot = str(row['uniprot']) if 'uniprot' in self.frame.columns else target
            if uniprot in seen_ids and seen_ids[uniprot] != seq:
                raise ValueError(f'{uniprot} has multiple sequence windows; use unique window cache IDs')
            seen_ids[uniprot] = seq
            key = (uniprot, seq, int(row['target_sequence_start']), int(row['target_sequence_end']))
            if key not in self.proteins:
                _, _, start, end = key
                if start < 0 or end - start != len(seq) or len(seq) == 0:
                    raise ValueError(f'{target}: expected zero-based, end-exclusive window of {len(seq)} residues')
                esm_path = Path(esm_dir) / f'{uniprot}.pt'
                emb = load_torch(esm_path)
                # New extractor embeds sequence metadata; legacy tensor caches remain readable.
                if isinstance(emb, dict):
                    if emb.get('sequence') != seq:
                        raise ValueError(f'{target}: ESM cache sequence differs from the CSV')
                    emb = emb['embedding']
                if not isinstance(emb, torch.Tensor) or tuple(emb.shape) != (len(seq), esm_dim):
                    raise ValueError(f'{target}: ESM cache must have shape {(len(seq), esm_dim)}')
                if not torch.isfinite(emb).all():
                    raise ValueError(f'{target}: ESM cache contains non-finite values')
                contact_path = Path(contact_dir) / f'{uniprot}.npy'
                contact = np.load(contact_path, allow_pickle=False)
                if contact.ndim != 2 or contact.shape[0] != contact.shape[1] or end > len(contact):
                    raise ValueError(f'{target}: contact map does not cover the requested window')
                crop = contact[start:end, start:end]
                if not np.isfinite(crop).all():
                    raise ValueError(f'{target}: non-finite contact map')
                edges = torch.from_numpy(np.vstack(np.where(crop > contact_threshold))).long()
                self.proteins[key] = (emb.float().detach(), edges)
                self.manifest['esm'][uniprot] = file_hash(esm_path)
                self.manifest['contacts'][uniprot] = file_hash(contact_path)
            if target in labels:
                label = torch.as_tensor(labels[target], dtype=torch.float32).flatten()
                if len(label) != len(seq) or not ((label == 0) | (label == 1)).all():
                    raise ValueError(f'{target}: pocket labels must be binary and window-aligned')
                self.labels[target] = label
            joint_key = (row['compound_iso_smiles'], key)
            if joint_key not in self.joints:
                self.joints[joint_key] = joint_graph(self.drugs[joint_key[0]], self.proteins[key][1], len(seq))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        sm, seq, target = row['compound_iso_smiles'], row['target_sequence'], str(row['target_key'])
        uniprot = str(row['uniprot']) if 'uniprot' in self.frame.columns else target
        key = (uniprot, seq, int(row['target_sequence_start']), int(row['target_sequence_end']))
        return {'drug_graph': self.drugs[sm], 'joint_graph': self.joints[(sm, key)],
                'protein_esm': self.proteins[key][0], 'smiles_tokens': self.tokens[sm],
                'pocket_label': self.labels.get(target, torch.zeros(len(seq))),
                'pocket_label_valid': target in self.labels, 'y': float(row['affinity'])}


def collate(samples):
    batch = {key: Batch.from_data_list([sample[key] for sample in samples])
             for key in ('drug_graph', 'joint_graph')}
    for key in ('protein_esm', 'smiles_tokens', 'pocket_label'):
        batch[key] = pad_sequence([sample[key] for sample in samples], batch_first=True)
    lengths = torch.tensor([len(sample['protein_esm']) for sample in samples])
    batch['protein_mask'] = torch.arange(batch['protein_esm'].shape[1])[None, :] < lengths[:, None]
    batch['smiles_mask'] = batch['smiles_tokens'].ne(0)
    batch['pocket_label_valid'] = torch.tensor([sample['pocket_label_valid'] for sample in samples])
    batch['y'] = torch.tensor([sample['y'] for sample in samples], dtype=torch.float32)
    return batch


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}
