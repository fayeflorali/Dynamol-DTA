import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import collate, to_device, load_torch
from .model import DynaMolDTA, pocket_loss
from .molunet.utils import get_deg_from_list
from . import frozen_split


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(labels, predictions):
    """CI via Fenwick counts; comparisons exclude tied labels and include prediction ties."""
    y, p = np.asarray(labels), np.asarray(predictions)
    _, ranks = np.unique(p, return_inverse=True)
    tree = np.zeros(len(np.unique(p)) + 1, dtype=np.int64)
    order = np.argsort(y)
    total, pairs, score, start = 0, 0, 0., 0

    def prefix(index):
        count = 0
        while index:
            count += int(tree[index])
            index -= index & -index
        return count

    while start < len(order):
        stop = start + 1
        while stop < len(order) and y[order[stop]] == y[order[start]]:
            stop += 1
        for index in order[start:stop]:
            rank = int(ranks[index]) + 1
            less, equal_or_less = prefix(rank - 1), prefix(rank)
            score += less + .5 * (equal_or_less - less)
            pairs += total
        for index in order[start:stop]:
            rank = int(ranks[index]) + 1
            while rank < len(tree):
                tree[rank] += 1
                rank += rank & -rank
        total += stop - start
        start = stop
    rm2 = None
    if len(y) > 1 and np.var(y) > 0 and np.var(p) > 0:
        yc, pc = y - y.mean(), p - p.mean()
        r2 = float(np.sum(yc * pc) ** 2 / (np.sum(yc ** 2) * np.sum(pc ** 2)))
        k = np.sum(y * p) / np.sum(p * p)
        r02 = 1 - np.sum((y - k * p) ** 2) / np.sum((y - y.mean()) ** 2)
        rm2 = float(r2 * (1 - np.sqrt(abs(r2 ** 2 - r02 ** 2))))
    return {'mse': float(np.mean((y - p) ** 2)),
            'ci': float(score / pairs) if pairs else None, 'rm2': rm2}


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    labels, predictions, losses = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction, auxiliary = model(batch, return_auxiliary=True)
            loss = torch.nn.functional.mse_loss(prediction, batch['y'])
            loss = loss + model.config['pocket_loss_weight'] * pocket_loss(batch, auxiliary)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training/evaluation loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
        labels.extend(batch['y'].detach().cpu().tolist())
        predictions.extend(prediction.detach().cpu().tolist())
        losses.extend([float(loss.detach())] * len(prediction))
    if not labels:
        raise ValueError('Empty data loader')
    result = regression_metrics(labels, predictions)
    result['loss'] = float(np.mean(losses))
    return result, labels, predictions


def loader_for(dataset, indices, batch_size, workers, training=False):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=training,
                      num_workers=workers, persistent_workers=workers > 0, collate_fn=collate)


def restore_model(path, device='cpu'):
    checkpoint = load_torch(path)
    if checkpoint.get('format_version') != 1 or checkpoint.get('model_name') != 'DynaMol-DTA':
        raise ValueError('Expected a DynaMol-DTA checkpoint; source-model checkpoints are not interchangeable')
    model = DynaMolDTA(checkpoint['config'], checkpoint['deg'], checkpoint['vocab_size']).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def train_folds(dataset, config, output_dir, split_dir, folds=5, fold=None,
                epochs=100, batch_size=32, workers=0, device='cpu', seed=0, lr=1e-3):
    if epochs < 1 or batch_size < 1 or workers < 0 or folds < 2:
        raise ValueError('epochs/batch-size must be positive; workers >= 0; folds >= 2')
    if len(dataset) < max(10, folds) or (fold is not None and not 0 <= fold < folds):
        raise ValueError('Need >=10 rows for the fixed 10% validation split and a valid fold index')
    if config['pocket_loss_weight'] < 0:
        raise ValueError('pocket_loss_weight must be nonnegative')
    if config['pocket_loss_weight'] > 0 and not dataset.labels:
        raise ValueError('Positive pocket loss weight requires annotated targets')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_split.SPLIT_DIR = str(Path(split_dir).resolve())
    results = []
    for fold_id in ([fold] if fold is not None else range(folds)):
        seed_everything(seed + fold_id)
        indices = frozen_split.load_or_create_fold_split(
            dataset, fold_id, folds, 'pair', str(dataset.csv))
        if any(len(values) == 0 for values in indices.values()):
            raise ValueError('All train/validation/test subsets must be nonempty')
        train_smiles = dataset.frame.iloc[indices['train']]['compound_iso_smiles'].unique()
        deg = get_deg_from_list([dataset.drugs[sm] for sm in train_smiles])
        vocab_size = max(46, max(dataset.vocab.stoi.values()) + 1)
        model = DynaMolDTA(config, deg, vocab_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loaders = {key: loader_for(dataset, value, batch_size, workers, key == 'train')
                   for key, value in indices.items()}
        checkpoint_path = output_dir / f'fold{fold_id}.pt'
        best = float('inf')
        history = []
        for epoch in range(1, epochs + 1):
            train_metrics, _, _ = run_epoch(model, loaders['train'], device, optimizer)
            valid_metrics, _, _ = run_epoch(model, loaders['val'], device)
            history.append({'epoch': epoch, 'train': train_metrics, 'val': valid_metrics})
            print(f"fold={fold_id} epoch={epoch} train_mse={train_metrics['mse']:.6f} "
                  f"val_mse={valid_metrics['mse']:.6f}", flush=True)
            if valid_metrics['mse'] < best:
                best = valid_metrics['mse']
                torch.save({'format_version': 1, 'model_name': 'DynaMol-DTA', 'config': config,
                            'state_dict': {key: value.detach().cpu().clone()
                                           for key, value in model.state_dict().items()},
                            'deg': deg.cpu(), 'vocab_size': vocab_size,
                            'data_manifest': dataset.manifest, 'fold': fold_id,
                            'splits': {key: value.tolist() for key, value in indices.items()},
                            'seed': seed, 'epoch': epoch, 'validation': valid_metrics}, checkpoint_path)
        # Test only the checkpoint selected by validation MSE.
        model, _ = restore_model(checkpoint_path, device)
        metrics, labels, predictions = run_epoch(model, loaders['test'], device)
        result = {'fold': fold_id, 'metrics': metrics, 'test_indices': indices['test'].tolist(),
                  'labels': labels, 'predictions': predictions, 'history': history}
        (output_dir / f'fold{fold_id}.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        results.append(result)
        del loaders
    summary = {'model': 'DynaMol-DTA', 'folds': results,
               'mean_mse': float(np.mean([item['metrics']['mse'] for item in results]))}
    (output_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary
