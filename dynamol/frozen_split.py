"""Frozen train/validation/test splits, materialised on disk.

``main.py`` and ``main_warm_up.py`` used to recompute the split on every run
from two hard-coded constants::

    KFold(n_splits=5, shuffle=True, random_state=0)     # -> the test fold
    random_split(..., generator=manual_seed(0))         # -> train / validation

That is reproducible only for as long as ``len(dataset)`` and the CSV row order
never change, and it left nothing on disk recording which row went where.  A
re-run that silently landed on a different split therefore looked exactly like
a re-run that did not -- changing ``DMFF_LIMIT``, or re-running the
preprocessing, was enough to move every row.

This module computes the same indices, writes them to
``splits/<dataset>_<mode>_fold<N>.json``, and loads them from that file on
every subsequent run.  Every script that trains -- the frozen-split copy
``DMFF-DTA-main1`` and the ``DMFF-DP1`` ablation -- reads the same file, so a
difference between their numbers cannot come from a different split.

The proportions are 70 / 10 / 20 of the whole table, with five folds::

    test    1 / n_folds of the table      6012 rows of 30056   (20.0%)
    val     10% of the whole table        3005 rows            (10.0%)
    train   the rest of the outer fold   21039 rows            (70.0%)

The 10% is taken from the whole table, not from the 80% left after the test
fold, because that is what the released ``DMFF-DTA-main`` computes
(``val_size = int(len(dataset) * 0.1)``).  Freezing the split may not change
what the baseline trains on.

One pair only.  An earlier revision of this file also froze a second pair --
10% of the outer fold, the 21640 / 2404 cut that ``DMFF-DP1`` had drifted to --
and each script picked whichever it happened to be written against.  Two
train/val pairs that are both "the validation set" is precisely the silent
divergence this module exists to prevent, so the second one is gone.  Changing
``n_folds`` away from 5 keeps the 10% rule but breaks the 70/20 reading.

``DMFF_SEED`` does not move the split.  The split is frozen at seed 0 on
purpose: varying the training seed over a fixed split is the comparison that
isolates model changes, and varying both at once is not.

Only the ungrouped ``pair`` mode is implemented.  ``splits.py`` still holds the
grouped modes (``cold_drug`` / ``cold_target`` / ``scaffold``), but nothing wires
them in any more, so ``split_mode`` here is a filename component and nothing
else -- see ``_check_split_mode`` for why every other value is refused rather
than quietly accepted.

Torch version note: ``randperm`` is what turns ``manual_seed(0)`` into row
indices, and its output is not guaranteed stable across torch releases.
Freezing the result removes that dependency from the experiment.
"""

import hashlib
import json
import os

import numpy as np
import torch
from sklearn.model_selection import KFold


SPLIT_DIR = os.environ.get('DMFF_SPLIT_DIR', './splits')
# Every list is stored in the order the inline code produced it, and training
# relies on that: ``train_loader`` shuffles, so ``RandomSampler``'s randperm is
# applied to whatever sequence the Subset yields.  Reordering ``train`` would
# reshuffle every batch and move the training trajectory, even though the set of
# rows is unchanged.  That is why ``train`` is stored rather than derived as
# ``outer_train - val``.
SPLIT_KEYS = ('train', 'val', 'test')
RECIPE = ('KFold(n_splits={n_folds}, shuffle=True, random_state=0), then '
          'randperm(generator=torch.Generator().manual_seed(0)) over the outer '
          'fold: train = prefix without the last 10% of the table, '
          'val = that last 10%')
SUPPORTED_SPLIT_MODES = ('pair',)


def _check_split_mode(split_mode):
    """Refuse a split mode that would be written but not honoured.

    ``compute_fold_split`` reproduces the ungrouped cut and never looks at
    ``split_mode``, so before this guard ``DMFF_SPLIT_MODE=cold_drug`` produced
    a file named ``..._cold_drug_fold0.json`` holding an ordinary random pair
    split.  No error, no warning, and a cold-start run that is not one -- the
    exact failure this module exists to make impossible.
    """
    if split_mode not in SUPPORTED_SPLIT_MODES:
        raise NotImplementedError(
            'DMFF_SPLIT_MODE={!r} is not implemented in frozen_split.py. The '
            'group modes live in splits.py::build_outer_splits, which the '
            'training scripts no longer call. Accepting {!r} here would write a '
            'random pair split into a file named after that mode, which reads '
            'as a cold-start run and is not one. Use one of {} -- or port the '
            'group modes into compute_fold_split() (GroupKFold / grouped '
            'random_split, same 70/10/20 and the same frozen file layout) '
            'before running under that label.'.format(
                split_mode, split_mode, list(SUPPORTED_SPLIT_MODES)
            )
        )


def _csv_fingerprint(csv_path):
    """md5 of the CSV the split is tied to, or None when the caller passed no
    path.  A path that does not exist is an error rather than a skipped check:
    silently dropping the comparison would let a mistyped path reuse another
    dataset's row indices."""
    if csv_path is None:
        return None
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            'Cannot fingerprint {}: no such file. Pass csv_path=None to '
            'disable the check deliberately.'.format(csv_path)
        )
    return _file_md5(csv_path)


def _file_md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sizes(n_rows, fold, n_folds):
    """Sizes the inline code produced, so a frozen file can be checked against
    them without recomputing the permutation."""
    # KFold hands the first (n_rows % n_folds) folds one extra row.
    test_size = n_rows // n_folds + (1 if fold < n_rows % n_folds else 0)
    outer_size = n_rows - test_size
    val_size = int(n_rows * 0.1)
    return {
        'test': test_size,
        'val': val_size,
        'train': outer_size - val_size,
    }


def _ratios(n_rows, sizes):
    return 'train {:.1f}% / val {:.1f}% / test {:.1f}%'.format(
        100.0 * sizes['train'] / n_rows,
        100.0 * sizes['val'] / n_rows,
        100.0 * sizes['test'] / n_rows,
    )


def compute_fold_split(n_rows, fold, n_folds):
    """Reproduce the indices the inline code produced, without touching disk."""
    if n_folds < 2:
        raise ValueError('n_folds must be at least 2, got {}'.format(n_folds))
    if not 0 <= fold < n_folds:
        raise ValueError('fold must be in [0, {}), got {}'.format(n_folds, fold))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    outer_train_indices, test_indices = list(kf.split(np.arange(n_rows)))[fold]

    outer_size = len(outer_train_indices)
    # The single randperm both cuts are taken from.  Order is kept exactly as
    # torch produced it -- the subsets are never sorted downstream, and there is
    # no reason for this file to be the one that reorders them.
    order = torch.randperm(
        outer_size, generator=torch.Generator().manual_seed(0)
    ).numpy()

    sizes = _expected_sizes(n_rows, fold, n_folds)
    if sizes['val'] >= outer_size:
        raise ValueError(
            'validation of {} rows does not fit in the {} outer training rows '
            'of fold {}'.format(sizes['val'], outer_size, fold)
        )

    return {
        'test': np.asarray(test_indices, dtype=np.int64),
        'train': np.asarray(
            outer_train_indices[order[:outer_size - sizes['val']]],
            dtype=np.int64,
        ),
        'val': np.asarray(
            outer_train_indices[order[outer_size - sizes['val']:]],
            dtype=np.int64,
        ),
    }


def dataset_key_from_path(csv_path):
    """Name the frozen file after the CSV it describes.

    Deriving the key from the data file rather than from a ``dataset_name``
    variable means ``main.py`` and the ``frozen_split.py`` CLI cannot disagree
    about which file to use -- both would otherwise have to remember that
    ``main.py`` appends ``_processed`` to the name partway through.  It also
    makes a kiba run unable to pick up davis's indices even before the CSV
    fingerprint is checked.
    """
    return os.path.splitext(os.path.basename(csv_path))[0]


def split_path(dataset_key, split_mode, fold):
    return os.path.join(
        SPLIT_DIR, '{}_{}_fold{}.json'.format(dataset_key, split_mode, fold)
    )


def _check(payload, path, n_rows, fold, n_folds, csv_path):
    """Reject a frozen file that does not describe the dataset on disk."""
    expected = {
        'n_rows': n_rows,
        'fold': fold,
        'n_folds': n_folds,
    }
    for key, want in expected.items():
        got = payload.get(key)
        if got != want:
            raise ValueError(
                'Frozen split {} says {}={!r} but the current run has {!r}. '
                'Refusing to reuse it -- rerun with DMFF_REFRESH_SPLIT=1 to '
                'regenerate, or point DMFF_SPLIT_DIR at the right directory.'
                .format(path, key, got, want)
            )

    want_md5 = _csv_fingerprint(csv_path)
    if want_md5 is not None:
        got_md5 = payload.get('csv_md5')
        if got_md5 is None:
            raise ValueError(
                'Frozen split {} carries no CSV fingerprint, so it cannot be '
                'matched against {}. Rerun with DMFF_REFRESH_SPLIT=1 to '
                'regenerate it.'.format(path, csv_path)
            )
        if got_md5 != want_md5:
            raise ValueError(
                'Frozen split {} was built from a CSV with md5 {} but {} now '
                'has md5 {}. The row indices in this file do not refer to the '
                'same rows any more.'.format(path, got_md5, csv_path, want_md5)
            )

    sizes = _expected_sizes(n_rows, fold, n_folds)
    arrays = {}
    for key in SPLIT_KEYS:
        raw = payload.get(key)
        if not isinstance(raw, list) or not raw:
            raise ValueError(
                'Frozen split {} has no usable {!r} list'.format(path, key)
            )
        array = np.asarray(raw, dtype=np.int64)
        if array.ndim != 1:
            raise ValueError('{!r} in {} is not one-dimensional'.format(key, path))
        if array.min() < 0 or array.max() >= n_rows:
            raise ValueError(
                '{!r} in {} references rows outside [0, {})'.format(
                    key, path, n_rows
                )
            )
        if len(array) != sizes[key]:
            raise ValueError(
                'Frozen split {} has {} {!r} indices, expected {} ({} of {} '
                'rows). A file holding a different val/train convention is the '
                'thing this check exists to catch -- do not "fix" it by hand, '
                'rerun with DMFF_REFRESH_SPLIT=1.'.format(
                    path, len(array), key, sizes[key], _ratios(n_rows, sizes),
                    n_rows
                )
            )
        arrays[key] = array

    # train / val / test must partition every row exactly once.
    partition = np.concatenate([arrays['train'], arrays['val'], arrays['test']])
    if len(np.unique(partition)) != n_rows:
        raise ValueError(
            'Frozen split {} does not partition the {} rows: train / val / '
            'test overlap or leave rows out.'.format(path, n_rows)
        )


def _dump(payload, path):
    """Write JSON with the metadata indented but the index lists on one line
    each -- a per-element indent turns a 300 KB file into a 5 MB one."""
    parts = []
    for key, value in payload.items():
        parts.append('  {}: {}'.format(
            json.dumps(key), json.dumps(value, separators=(',', ':'))
        ))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('{\n' + ',\n'.join(parts) + '\n}\n')


def freeze_all_folds(n_rows, n_folds, dataset_key, split_mode, csv_path):
    """Write the frozen file for every fold. Returns the paths written."""
    _check_split_mode(split_mode)
    written = []
    for fold in range(n_folds):
        path = split_path(dataset_key, split_mode, fold)
        computed = compute_fold_split(n_rows, fold, n_folds)
        _write(computed, path, n_rows, fold, n_folds, dataset_key, split_mode,
               csv_path)
        written.append(path)
    return written


def _write(computed, path, n_rows, fold, n_folds, dataset_key, split_mode,
           csv_path):
    payload = {
        'dataset': dataset_key,
        'split_mode': split_mode,
        'fold': fold,
        'n_folds': n_folds,
        'n_rows': n_rows,
        'csv_path': csv_path,
        'csv_md5': _csv_fingerprint(csv_path),
        'recipe': RECIPE.format(n_folds=n_folds),
    }
    for key in SPLIT_KEYS:
        payload[key] = computed[key].tolist()
    _dump(payload, path)


def load_or_create_fold_split(dataset, fold, n_folds, split_mode, csv_path,
                              dataset_key=None, verbose=True):
    """Return ``{'train', 'val', 'test'}`` as int64 arrays.

    Loads the frozen indices if they exist and still describe this dataset;
    otherwise computes them the old way and freezes the result.  Set
    ``DMFF_REFRESH_SPLIT=1`` to recompute and overwrite an existing file.
    ``dataset_key`` defaults to the CSV's file stem.
    """
    _check_split_mode(split_mode)
    n_rows = len(dataset)
    if dataset_key is None:
        dataset_key = dataset_key_from_path(csv_path)
    path = split_path(dataset_key, split_mode, fold)
    refresh = os.environ.get('DMFF_REFRESH_SPLIT') == '1'

    if os.path.exists(path) and not refresh:
        with open(path) as handle:
            payload = json.load(handle)
        _check(payload, path, n_rows, fold, n_folds, csv_path)
        if verbose:
            sizes = _expected_sizes(n_rows, fold, n_folds)
            print('Frozen split loaded: {} (rows {}, train {} / val {} / '
                  'test {} -- {})'.format(
                      path,
                      n_rows,
                      sizes['train'],
                      sizes['val'],
                      sizes['test'],
                      _ratios(n_rows, sizes),
                  ))
        return {key: np.asarray(payload[key], dtype=np.int64)
                for key in SPLIT_KEYS}

    computed = compute_fold_split(n_rows, fold, n_folds)
    _write(computed, path, n_rows, fold, n_folds, dataset_key, split_mode,
           csv_path)
    if verbose:
        sizes = _expected_sizes(n_rows, fold, n_folds)
        print('Frozen split {}: {} (rows {}, train {} / val {} / test {} -- '
              '{})'.format(
                  'rewritten' if refresh else 'created',
                  path,
                  n_rows,
                  sizes['train'],
                  sizes['val'],
                  sizes['test'],
                  _ratios(n_rows, sizes),
              ))
    return computed


def _count_rows(csv_path):
    """Row count via the csv module, so quoted fields containing newlines do
    not inflate it the way a raw line count would."""
    import csv

    with open(csv_path, newline='') as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main():
    """``python frozen_split.py [--folds N]`` -- freeze every fold for the
    dataset this directory holds, without starting a training run."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rows', type=int, default=None,
                        help='row count; defaults to counting the CSV')
    parser.add_argument('--folds', type=int,
                        default=int(os.environ.get('DMFF_NUM_FOLDS', '5')))
    parser.add_argument('--dataset', default='davis',
                        help='dataset stem; resolved with the same '
                             'resolve_processed_data_path() main.py uses')
    parser.add_argument('--split-mode', default='pair')
    parser.add_argument('--csv', default=None,
                        help='overrides --dataset; defaults to whatever '
                             'resolve_processed_data_path() finds')
    args = parser.parse_args()

    from utils import resolve_processed_data_path

    csv_path = args.csv or resolve_processed_data_path(args.dataset)
    rows = args.rows if args.rows is not None else _count_rows(csv_path)
    dataset_key = dataset_key_from_path(csv_path)

    written = freeze_all_folds(rows, args.folds, dataset_key,
                               args.split_mode, csv_path)
    for fold in range(args.folds):
        sizes = _expected_sizes(rows, fold, args.folds)
        print('fold {}: train {} / val {} / test {}  ({})'.format(
            fold, sizes['train'], sizes['val'], sizes['test'],
            _ratios(rows, sizes)
        ))
    print('Froze {} folds from {} ({} rows, key {!r})'.format(
        len(written), csv_path, rows, dataset_key
    ))
    for path in written:
        print('  ' + path)


if __name__ == '__main__':
    main()
