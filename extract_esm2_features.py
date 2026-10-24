"""
Offline ESM-2 feature extraction for DMFF-DTA's protein branch.

Runs esm2_t30_150M_UR50D (fair-esm) over every unique protein sequence in
`{dataset_name}_processed.csv` and caches per-residue embeddings to
`./esm2_embedding_{dataset_name}/{uniprot}.pt`, mirroring the existing
`target_contact_map_{dataset_name}/{uniprot}.npy` layout.

Note: `target_sequence` in the processed CSV is already the binding-site
window (e.g. 300 aa) sliced out of the full UniProt sequence via
`target_sequence_start`/`target_sequence_end` — the full-length sequence
text itself isn't shipped anywhere in this repo (only the full-length
contact map, which has no residue identities). So this script embeds the
window sequence directly rather than the full UniProt sequence; the
resulting embedding is already aligned 1:1 with `target_sequence` and needs
no further start:end slicing at train time.

Each cached file holds sequence/model metadata and a float32 embedding of shape [window_len, 640] where
row i corresponds to residue i of `target_sequence` (the ESM-2 <cls>/<eos>
tokens are stripped before saving).

Usage:
    python extract_esm2_features.py --dataset davis
    python extract_esm2_features.py --dataset kiba --device cuda:0

Requires: pip install fair-esm
"""
import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

ESM2_MODEL = 'esm2_t30_150M_UR50D'
ESM2_LAYER = 30
ESM2_EMBED_DIM = 640
MAX_TOKENS_PER_CHUNK = 1022  # ESM-2 positional limit is 1024 incl. <cls>/<eos>


def load_targets(dataset_name, csv_path=None):
    """Map uniprot -> binding-site window sequence (target_sequence)."""
    df = pd.read_csv(csv_path or f'./{dataset_name}_processed.csv')
    uniprot_to_seq = {}
    for i in range(len(df)):
        uniprot = df.loc[i, 'target_key'] if dataset_name == 'kiba' else df.loc[i, 'uniprot']
        seq = df.loc[i, 'target_sequence']
        if uniprot in uniprot_to_seq and uniprot_to_seq[uniprot] != seq:
            raise ValueError(
                f"UniProt id {uniprot!r} maps to two different target_sequence windows; "
                "the ESM-2 cache assumes a 1:1 uniprot->window mapping."
            )
        uniprot_to_seq[uniprot] = seq
    return uniprot_to_seq


@torch.no_grad()
def embed_sequence(model, batch_converter, seq, device):
    """Embed a single sequence, chunking if it exceeds ESM-2's context window.

    Binding-site windows in this dataset (~300 aa) are well within ESM-2's
    1024-token limit, so the chunking path is a safety net for unexpectedly
    long sequences rather than something normally exercised.
    """
    if len(seq) <= MAX_TOKENS_PER_CHUNK:
        _, _, tokens = batch_converter([('target', seq)])
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[ESM2_LAYER], return_contacts=False)
        reps = out['representations'][ESM2_LAYER][0]  # [seq_len+2, 640]
        return reps[1:len(seq) + 1].float().cpu()  # strip <cls>/<eos>

    chunks = []
    for start in range(0, len(seq), MAX_TOKENS_PER_CHUNK):
        sub = seq[start:start + MAX_TOKENS_PER_CHUNK]
        _, _, tokens = batch_converter([('target', sub)])
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[ESM2_LAYER], return_contacts=False)
        reps = out['representations'][ESM2_LAYER][0]
        chunks.append(reps[1:len(sub) + 1].float().cpu())
    return torch.cat(chunks, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['davis', 'kiba'])
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--csv', help='path to the processed CSV')
    parser.add_argument('--out-dir', '--out_dir', dest='out_dir', default=None,
                         help='defaults to ./esm2_embedding_{dataset}/')
    parser.add_argument('--overwrite', action='store_true',
                         help='re-embed even if a cached .pt file already exists')
    args = parser.parse_args()

    out_dir = args.out_dir or f'./esm2_embedding_{args.dataset}/'
    os.makedirs(out_dir, exist_ok=True)

    import esm
    print(f'Loading {ESM2_MODEL} ...')
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM2_MODEL)
    model = model.to(args.device).eval()
    batch_converter = alphabet.get_batch_converter()

    uniprot_to_seq = load_targets(args.dataset, args.csv)
    print(f'{len(uniprot_to_seq)} unique UniProt targets to embed.')

    for uniprot, seq in tqdm(uniprot_to_seq.items()):
        out_path = os.path.join(out_dir, f'{uniprot}.pt')
        if os.path.exists(out_path) and not args.overwrite:
            continue
        emb = embed_sequence(model, batch_converter, seq, args.device)
        assert emb.shape[0] == len(seq), (
            f'{uniprot}: embedded {emb.shape[0]} residues but window sequence has {len(seq)}'
        )
        torch.save({'embedding': emb, 'sequence': seq, 'model': ESM2_MODEL}, out_path)

    print(f'Done. Embeddings cached in {out_dir}')


if __name__ == '__main__':
    main()
