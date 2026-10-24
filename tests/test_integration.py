import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from dynamol.data import AffinityDataset, collate, tokenize
from dynamol.model import DynaMolDTA, pocket_loss
from dynamol.engine import regression_metrics, train_folds, restore_model, loader_for, run_epoch
from dynamol.molunet.utils import get_deg_from_list
from dynamol.smoke_data import create_fixture

ROOT = Path(__file__).resolve().parents[1]


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        csv, esm, contacts = create_fixture(self.root)
        self.data = AffinityDataset(csv, esm, contacts, ROOT / 'Vocab/smiles_vocab.pkl',
                                    pocket_labels=self.root / 'pocket_labels.npz')
        self.config = json.loads((ROOT / 'config.json').read_text())
        self.deg = get_deg_from_list(list(self.data.drugs.values()))

    def test_three_modules_receive_gradients(self):
        model = DynaMolDTA(self.config, self.deg).train()
        batch = collate([self.data[0], self.data[1]])
        prediction, auxiliary = model(batch, True)
        loss = (prediction - batch['y']).square().mean()
        loss.backward()
        for name, module in [('ESM projection', model.protein_encoder),
                             ('MolUNet', model.molunet), ('dynamic pocket', model.selector)]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, name)
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertTrue(torch.equal(auxiliary['pocket_scores'].detach().round(),
                                    auxiliary['hard_pocket_scores']))
        self.assertFalse(auxiliary['pocket_scores'][~batch['protein_mask']].any())
        # Reordering molecules must not change their predictions in evaluation mode.
        model.eval()
        with torch.no_grad():
            expected = model(batch)
            reordered = model(collate([self.data[1], self.data[0]]))
        torch.testing.assert_close(reordered.flip(0), expected, rtol=1e-4, atol=1e-5)

    def test_singleton_modes_and_supervision(self):
        for mode in ['dynamic_pocket', 'static_sparse', 'full_online']:
            config = copy.deepcopy(self.config)
            config['pocket_mode'] = mode
            config['pocket_loss_weight'] = 1.
            model = DynaMolDTA(config, self.deg).train()
            batch = collate([self.data[0]])
            prediction, auxiliary = model(batch, True)
            self.assertEqual(tuple(prediction.shape), (1,))
            loss = (prediction - batch['y']).square().mean() + pocket_loss(batch, auxiliary)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(tokenize('C' * 540, self.data.vocab)), 542)

    def test_training_checkpoint_and_spawn(self):
        summary = train_folds(self.data, self.config, self.root / 'run', self.root / 'splits',
                              folds=2, epochs=1, batch_size=2)
        self.assertEqual(len(summary['folds']), 2)
        model, checkpoint = restore_model(self.root / 'run/fold0.pt')
        self.assertEqual(checkpoint['data_manifest'], self.data.manifest)
        _, _, predictions = run_epoch(model, loader_for(self.data, checkpoint['splits']['test'], 2, 0), 'cpu')
        np.testing.assert_array_equal(predictions, summary['folds'][0]['predictions'])
        loader = DataLoader(self.data, batch_size=2, num_workers=2, collate_fn=collate,
                            multiprocessing_context='spawn')
        self.assertEqual(sum(len(batch['y']) for batch in loader), len(self.data))

    def test_ci_with_ties(self):
        self.assertEqual(regression_metrics([3, 1, 2], [3, 1, 2])['ci'], 1.)
        self.assertEqual(regression_metrics([3, 1, 2], [1, 3, 2])['ci'], 0.)
        self.assertEqual(regression_metrics([3, 1, 2], [1, 1, 1])['ci'], .5)
        self.assertIsNone(regression_metrics([1], [2])['ci'])


if __name__ == '__main__':
    unittest.main()
