"""Train DynaMol-DTA using precomputed ESM-2 windows and full contact maps."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
from dynamol.data import AffinityDataset
from dynamol.engine import train_folds

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--esm-dir', required=True)
    parser.add_argument('--contact-dir', required=True)
    parser.add_argument('--vocab', default=str(ROOT / 'Vocab/smiles_vocab.pkl'))
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--pocket-labels')
    parser.add_argument('--output', default=str(ROOT / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    parser.add_argument('--split-dir', default=str(ROOT / 'splits'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    dataset = AffinityDataset(args.csv, args.esm_dir, args.contact_dir, args.vocab,
                              esm_dim=config['esm_dim'], pocket_labels=args.pocket_labels)
    result = train_folds(dataset, config, args.output, args.split_dir, args.folds, args.fold,
                          args.epochs, args.batch_size, args.workers, args.device, args.seed, args.lr)
    print(f"mean_test_mse={result['mean_mse']:.6f}; output={args.output}")


if __name__ == '__main__':
    main()
