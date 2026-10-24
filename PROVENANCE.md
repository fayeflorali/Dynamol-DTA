# Module provenance

This directory combines the group's three local DMFF-DTA variants. Source projects
were read/copied, not modified while creating DynaMol-DTA.

| Local source | Destination | Use |
|---|---|---|
| DMFF-DP1/dynamic_pocket.py | dynamol/dynamic_pocket.py | Ligand-conditioned straight-through residue gate |
| DMFF-DP1/model.py | dynamol/dmff_layers.py | GIN and LinkAttention; original helper classes retained |
| DMFF-DP1/frozen_split.py | dynamol/frozen_split.py | Fixed pair-fold indices, validation and fingerprint checks |
| DMFF-DP1/build_vocab.py, Vocab/smiles_vocab.pkl | dynamol/build_vocab.py, Vocab/ | Existing DMFF token vocabulary |
| DMFF-MolUNet-Fusion-main/models/MolUnet.py, layers.py, utils.py | dynamol/molunet/ | Multiscale atom encoder, including previously corrected scatter compatibility |
| DMFF-DTA-ALDH2-perf-dataloader-mask-fix/.../extract_esm2_features.py | extract_esm2_features.py | ESM-2 window extraction, extended with explicit CSV path and cache sequence metadata |

The main forward model, dual-graph data interface and training engine are new
integration code. No UnetDTI ProteinCNN/BAN or old top-level fusion head is copied
into the active model. No fixed trainable node-feature cache is used.

LICENSE is copied from DMFF-DP1. Original source headers in the MolUNet files are
retained; the local MolUNet source directory did not contain a standalone LICENSE.
Retain upstream attribution and confirm the original MolUNet licensing before
external redistribution.

The model is based on DMFF-DTA; cite the original paper when reporting results:
He et al., "Dual modality feature fused neural network integrating binding site
information for drug target affinity prediction", npj Digital Medicine 8, 67 (2025).
Also credit the original MolUNet and ESM-2 work when publishing results.
