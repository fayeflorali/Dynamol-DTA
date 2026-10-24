# Initial integration validation

Validated locally on CPU with Python from `C:/Users/11392/miniconda3/envs/dmff-cpu`:

- torch 1.12.1+cpu, torch-geometric 2.3.1, RDKit 2022.03.2
- NumPy 1.26.4, pandas 2.3.1, SciPy 1.10.1, scikit-learn 1.3.0

Completed checks:

1. All 19 Python files parse successfully.
2. `python -m unittest tests.test_integration -v`: four tests passed.
   - Finite, nonzero gradients reach the ESM projection, MolUNet, and pocket selector.
   - Single-sample forward/backward works for all three pocket modes.
   - Padding masks, batch reordering, long token sequences, and tied CI scores are checked.
   - Two folds train and save separately; restoring the checkpoint reproduces saved test predictions.
   - Two DataLoader workers using Windows spawn load all fixture rows.
3. `python smoke_test.py --output runs/smoke_verified`: passed.
4. `train.py` was run for one CPU epoch on the same synthetic fixture with a
   previously frozen split; output is in `runs/cli_verified`.
5. `evaluate.py` loaded `runs/smoke_verified/fold0.pt`, checked the input manifest,
   and evaluated its saved test fold successfully.

The fixture contains ten artificial affinity rows, real RDKit molecular graphs,
artificial protein contact maps and random tensors standing in for cached ESM
embeddings. It validates software integration, not biological accuracy.
No real-data Davis/KIBA training, ESM-2 model download/extraction, CUDA execution,
or external benchmark comparison was performed.

During integration, the MolUNet query dimension was aligned with its configured
FP query width. The one-dimensional rm2 formula uses elementwise sums rather
than a covariance matrix; this avoids a local NumPy/BLAS stall observed during
testing while computing the same statistic.
