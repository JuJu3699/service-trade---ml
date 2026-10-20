import unittest
import tempfile
from pathlib import Path
import numpy as np
import yaml
from src.generate_demo_data import generate
from src.runner import (load_config, read_model_ready_data, design_matrices,
                        fold_diagnostics, ShuffledGroupKFold, make_nuisance_regressor)

class TestDataAndSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(Path("config.yaml"))
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "fake.csv"
        generate(cls.path, n_partners=12)
        cls.df = read_model_ready_data(cls.path, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_period_counts(self):
        self.assertEqual(len(self.df), 12 * 20)
        self.assertEqual(len(self.df[self.df.year <= 2019]), 12 * 15)
        self.assertEqual(len(self.df[self.df.year >= 2020]), 12 * 5)

    def test_baseline_matrices(self):
        d = self.df[self.df.year >= 2020].reset_index(drop=True)
        y,t,z,w,groups,names = design_matrices(d,self.cfg,"travel_import","direct_cities","baseline")
        self.assertEqual(t.ndim,1)
        self.assertEqual(z.ndim,1)
        self.assertEqual(names,["ln_direct_cities"])
        self.assertEqual(w.shape[0],60)

    def test_interaction_instrument_product(self):
        d = self.df[self.df.year >= 2020].reset_index(drop=True)
        y,t,z,w,groups,names = design_matrices(d,self.cfg,"transport_export","avg_seats","interaction")
        self.assertEqual(t.shape,(60,3))
        self.assertEqual(z.shape,(60,3))
        np.testing.assert_allclose(t[:,2],t[:,0]*t[:,1])
        np.testing.assert_allclose(z[:,2],z[:,0]*z[:,1])
        diag = fold_diagnostics(groups,self.cfg,42,0)
        self.assertEqual(diag['n_folds'],self.cfg['folds'])

    def test_random_group_splits(self):
        g = self.df[self.df.year >= 2020]["partner"].to_numpy()
        a = list(ShuffledGroupKFold(3, 1).split(np.ones((len(g),1)), groups=g))
        b = list(ShuffledGroupKFold(3, 2).split(np.ones((len(g),1)), groups=g))
        self.assertTrue(any(not np.array_equal(i[1],j[1]) for i,j in zip(a,b)))
        for train,test in a:
            self.assertFalse(set(g[train])&set(g[test]))

    def test_all_learner_models_construct(self):
        for learner in self.cfg['learners']:
            for count in (1,3):
                self.assertIsNotNone(make_nuisance_regressor(learner,self.cfg,42,count))

if __name__ == '__main__':
    unittest.main()

class TestOutputFlowWithoutEconML(unittest.TestCase):
    """I/O integration only: mock estimator output, not an EconML numerical test."""
    def test_reports_written_with_mock_coefficient_provider(self):
        from unittest.mock import patch
        from src.runner import run
        cfg = load_config(Path("config.yaml"))
        cfg = dict(cfg)
        cfg["outcomes"] = ["travel_import"]
        cfg["models"] = ["baseline", "interaction"]
        cfg["learners"] = ["rf", "lasso"]
        cfg["split_repetitions"] = 2
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "fake.csv"
            generate(path, 8)
            def fake_fit(y,t,z,w,groups,cfg,learner,seed,repeat_index):
                nt = t.shape[1] if t.ndim == 2 else 1
                return np.arange(1,nt+1,dtype=float) * .01 + .001*repeat_index
            with patch("src.runner.fit_econml_dmliv", side_effect=fake_fit):
                run(cfg, path, Path(dirname)/"out")
            out = Path(dirname)/"out"
            self.assertTrue((out/"comparison_by_learner.csv").exists())
            summary = __import__("pandas").read_csv(out/"comparison_by_learner.csv")
            self.assertIn("rf", summary.columns)
            self.assertIn("lasso", summary.columns)
            self.assertEqual(len(summary), (2*2*1+2*2*3))
