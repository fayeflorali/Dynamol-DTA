import math

import torch
import torch.nn as nn


class DynamicPocketSelectorLite(nn.Module):
    """Ligand-conditioned neural residue gate with straight-through binarization."""

    def __init__(self, residue_dim, atom_dim, hidden_dim=128, n_heads=4,
                 temperature=0.5, target_ratio=0.15,
                 selection_threshold=None):
        super(DynamicPocketSelectorLite, self).__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        # target_ratio is an initialization prior, not a fixed selection
        # budget. Training is free to move this bias from its initial value.
        prior_logit = math.log(target_ratio / (1.0 - target_ratio)) \
            if target_ratio < 1.0 else 10.0

        # Soft scores are sigmoid(centered_logits / temperature + gate_bias),
        # and centered_logits is zero-mean by construction. The threshold that
        # puts the *average* residue exactly on the decision boundary is
        # therefore sigmoid(gate_bias) -- the prior itself.
        #
        # Note what that makes the binarization rate at initialization: about
        # half the residues, not `target_ratio`. target_ratio sets where the
        # *soft* scores sit (their mean is ~0.15 at the default) and where the
        # boundary falls; the hard count is a consequence of both, and training
        # moves it. This is deliberately not a top-k rule -- see the README's
        # "the count is learned" design note. If a 15% *hard* rate at init is
        # ever wanted, the threshold has to become a quantile, which is top-k
        # wearing a disguise.
        #
        # The previous default of a flat 0.5 broke the alignment: the boundary
        # sat at centered_logits = -temperature * gate_bias, roughly +0.87 at
        # init, against a measured logit spread of only ~0.64. No residue could
        # ever cross it, so the gate started fully closed (0% selected) and the
        # selector could only open it by driving the global bias up.
        if selection_threshold is None:
            selection_threshold = torch.sigmoid(
                torch.tensor(prior_logit)
            ).item()
        if not 0.0 < selection_threshold < 1.0:
            raise ValueError("selection_threshold must be in (0, 1)")

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.temperature = float(temperature)
        self.target_ratio = float(target_ratio)
        self.selection_threshold = float(selection_threshold)
        # The hard decision is `soft_scores >= selection_threshold`, i.e.
        # `gate_logits >= logit(selection_threshold)`. Subtracting that
        # threshold turns the decision into a plain sign test, which is what
        # the pocket supervision (utils.pocket_supervision_loss) needs: its
        # positive class has to sit exactly where the gate actually flips,
        # not at some nearby value. Supervising raw gate_logits instead would
        # anchor the positives at gate_logits = 0 while the gate flips at
        # gate_bias, and the two drift apart as soon as gate_bias learns
        # anything.
        self.selection_logit = math.log(
            self.selection_threshold / (1.0 - self.selection_threshold)
        )
        self.gate_bias = nn.Parameter(torch.tensor(prior_logit))
        self.residue_query = nn.Sequential(
            nn.LayerNorm(residue_dim),
            nn.Linear(residue_dim, hidden_dim, bias=False),
        )
        self.atom_key = nn.Sequential(
            nn.LayerNorm(atom_dim),
            nn.Linear(atom_dim, hidden_dim, bias=False),
        )
        self.static_atom = nn.Parameter(torch.empty(1, 1, atom_dim))
        nn.init.normal_(self.static_atom, std=0.02)

    def _centered_logits(self, residue_features, atom_features, residue_mask,
                         atom_mask):
        """Residue/atom compatibility, centered so its mean over the window is 0.

        Split out of forward() because it is evaluated twice per step when
        pocket supervision is on -- see the `supervised` argument there.
        """
        batch_size, residue_count = residue_mask.shape
        query = self.residue_query(residue_features).view(
            batch_size, residue_count, self.n_heads, self.head_dim
        ).transpose(1, 2)
        key = self.atom_key(atom_features).view(
            batch_size, atom_features.shape[1], self.n_heads, self.head_dim
        ).transpose(1, 2)

        compatibility = torch.einsum("bhld,bhad->bhla", query, key)
        compatibility = compatibility / math.sqrt(self.head_dim)
        compatibility = compatibility.masked_fill(
            ~atom_mask[:, None, None, :], -30.0
        )
        atom_count = atom_mask.sum(dim=1).clamp_min(1).to(compatibility.dtype)
        logits = torch.logsumexp(compatibility, dim=-1)
        logits = logits - atom_count.log()[:, None, None]
        logits = logits.mean(dim=1)
        logits = logits.masked_fill(~residue_mask, -30.0)

        valid = residue_mask.to(logits.dtype)
        valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (logits * valid).sum(dim=1, keepdim=True) / valid_count
        centered = (logits - mean).masked_fill(~residue_mask, -30.0)
        return logits, centered

    def forward(self, residue_features, atom_features, residue_mask, atom_mask,
                ligand_conditioned=True, supervised=False):
        if residue_features.dim() != 3 or atom_features.dim() != 3:
            raise ValueError("residue_features and atom_features must be 3D")

        residue_mask = residue_mask.bool()
        atom_mask = atom_mask.bool()
        batch_size, residue_count = residue_mask.shape
        if not ligand_conditioned:
            atom_features = self.static_atom.expand(batch_size, -1, -1)
            atom_mask = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=atom_features.device
            )
        if atom_features.shape[:2] != atom_mask.shape:
            raise ValueError("atom embeddings and atom mask are not aligned")

        valid = residue_mask.to(residue_features.dtype)
        logits, centered_logits = self._centered_logits(
            residue_features, atom_features, residue_mask, atom_mask
        )
        gate_logits = centered_logits / self.temperature + self.gate_bias
        # Positive exactly where the hard gate selects. Unmasked on purpose:
        # the caller weights each position by residue_mask, and a BCE target
        # wants the real logit rather than a -30 sentinel.
        #
        # `supervised` decides which graph this tensor carries. The pocket
        # label describes the target, not the ligand, so it has no business
        # reaching the residue or atom encoders -- that would make the ablation
        # depend on a backbone that is slated for replacement. Recomputing from
        # *detached inputs* cuts exactly that, while leaving this module's own
        # parameters (both projections, and gate_bias) fully trainable. It
        # costs one extra compatibility pass, so it is skipped when there is
        # no supervision to apply.
        if supervised:
            # The residue side is always external to this module, so it is
            # always cut. The atom side is external only when it came from the
            # ligand: in static_sparse it *is* self.static_atom, a parameter of
            # this module and the only thing that arm has to learn its pocket
            # from, so detaching that would leave the gate with nothing but
            # gate_bias. Cut by provenance, not by tensor.
            detached_atoms = (atom_features.detach() if ligand_conditioned
                              else atom_features)
            _, centered_for_supervision = self._centered_logits(
                residue_features.detach(), detached_atoms, residue_mask,
                atom_mask,
            )
            decision_logits = (
                centered_for_supervision / self.temperature + self.gate_bias
                - self.selection_logit
            )
        else:
            # Unused: utils.pocket_supervision_loss returns None without a
            # positive weight. Kept on the live graph so the unsupervised path
            # stays byte-for-byte the computation it was before.
            decision_logits = gate_logits - self.selection_logit
        soft_scores = torch.sigmoid(gate_logits) * valid
        hard_scores = (
            (soft_scores >= self.selection_threshold).to(soft_scores.dtype)
            * valid
        )
        # Exact binary gates in the forward pass; sigmoid gradients in the
        # backward pass. Evaluation uses the same binary decision boundary.
        pocket_scores = (
            hard_scores + soft_scores - soft_scores.detach()
            if self.training else hard_scores
        )

        return {
            "pocket_logits": logits,
            "gate_logits": gate_logits,
            "decision_logits": decision_logits,
            "soft_pocket_scores": soft_scores,
            "hard_pocket_scores": hard_scores,
            "pocket_scores": pocket_scores,
            "pocket_mask": residue_mask,
        }
