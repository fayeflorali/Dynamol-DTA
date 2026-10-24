import argparse
import json
from pathlib import Path
import torch

from dynamol.data import AffinityDataset
from dynamol.engine import restore_model, loader_for, run_epoch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Evaluate the saved test fold with identical input assets')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--esm-dir', required=True)
    parser.add_argument('--contact-dir', required=True)
    parser.add_argument('--vocab', default=str(ROOT / 'Vocab/smiles_vocab.pkl'))
    parser.add_argument('--pocket-labels')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, checkpoint = restore_model(args.checkpoint, args.device)
    dataset = AffinityDataset(args.csv, args.esm_dir, args.contact_dir, args.vocab,
                              esm_dim=checkpoint['config']['esm_dim'], pocket_labels=args.pocket_labels)
    if dataset.manifest != checkpoint['data_manifest']:
        raise ValueError('CSV, vocabulary, ESM cache, labels or contact maps differ from training')
    loader = loader_for(dataset, checkpoint['splits']['test'], args.batch_size, 0)
    metrics, _, _ = run_epoch(model, loader, args.device)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
