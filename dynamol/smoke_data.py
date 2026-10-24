"""Synthetic inputs for software checks only, never benchmark results."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def create_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    esm_dir, contact_dir = directory / 'esm', directory / 'contacts'
    esm_dir.mkdir(exist_ok=True)
    contact_dir.mkdir(exist_ok=True)
    generator = torch.Generator().manual_seed(17)
    sequences = {'P1': 'ACDEFG', 'P2': 'LMNPQRST'}
    labels = {}
    for name, sequence in sequences.items():
        torch.save({'sequence': sequence,
                    'embedding': torch.randn(len(sequence), 640, generator=generator)}, esm_dir / f'{name}.pt')
        matrix = np.eye(len(sequence) + 4, dtype=np.float32)
        matrix[2, 3] = matrix[3, 2] = 1
        np.save(contact_dir / f'{name}.npy', matrix)
        labels[name] = np.array([1, 0] * (len(sequence) // 2), dtype=np.float32)
    rows = []
    for i in range(10):
        target = 'P1' if i % 2 == 0 else 'P2'
        sequence = sequences[target]
        rows.append(dict(compound_iso_smiles=['CCO', 'CCN', 'CC(=O)O'][i % 3],
                         target_key=target, uniprot=target, target_sequence=sequence,
                         target_sequence_start=2, target_sequence_end=2 + len(sequence),
                         affinity=5.0 + i * .1))
    csv = directory / 'synthetic.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    np.savez(directory / 'pocket_labels.npz', **labels)
    return csv, esm_dir, contact_dir
