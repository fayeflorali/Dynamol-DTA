"""Train and restore one tiny fold with synthetic ESM inputs (not an accuracy benchmark)."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
from dynamol.data import AffinityDataset
from dynamol.engine import train_folds, restore_model, loader_for, run_epoch
from dynamol.smoke_data import create_fixture

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs' / ('smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    args = parser.parse_args()
    torch.set_num_threads(2)
    csv, esm, contacts = create_fixture(args.output / 'inputs')
    config = json.loads((ROOT / 'config.json').read_text())
    dataset = AffinityDataset(csv, esm, contacts, ROOT / 'Vocab/smiles_vocab.pkl')
    summary = train_folds(dataset, config, args.output, args.output / 'splits',
                          folds=2, fold=0, epochs=1, batch_size=2)
    model, checkpoint = restore_model(args.output / 'fold0.pt')
    loader = loader_for(dataset, checkpoint['splits']['test'], 2, 0)
    _, _, predictions = run_epoch(model, loader, 'cpu')
    np.testing.assert_array_equal(predictions, summary['folds'][0]['predictions'])
    print('PASS: train/validation/test, singleton final batch, save/load and exact prediction replay')
    print('Synthetic ESM inputs only; these metrics are NOT real-data model performance.')
    print('Output:', args.output)


if __name__ == '__main__':
    main()
