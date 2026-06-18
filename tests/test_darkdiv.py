import unittest
import numpy as np
import pandas as pd
import torch
import sys
import os

# Add src/ to path so we can import pmf_dark
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from pmf_dark.darkdiv import (
    infer_y_type,
    prepare_data,
    compute_predictions,
    compute_dark_diversity,
    PMFDark,
)


class TestDarkDiv(unittest.TestCase):
    def setUp(self):
        # Create small mock datasets for testing
        self.n_sites = 20
        self.n_species = 5
        self.n_env = 2

        # Binary data
        self.y_binary = pd.DataFrame(
            np.random.randint(0, 2, size=(self.n_sites, self.n_species)),
            index=[f"site_{i}" for i in range(self.n_sites)],
            columns=[f"spec_{j}" for j in range(self.n_species)],
        )

        # Count data
        self.y_count = pd.DataFrame(
            np.random.randint(0, 10, size=(self.n_sites, self.n_species)),
            index=[f"site_{i}" for i in range(self.n_sites)],
            columns=[f"spec_{j}" for j in range(self.n_species)],
        )

        # Env predictors
        self.x = pd.DataFrame(
            np.random.randn(self.n_sites, self.n_env),
            index=[f"site_{i}" for i in range(self.n_sites)],
            columns=[f"env_{j}" for j in range(self.n_env)],
        )

    def test_infer_y_type_presence_absence(self):
        # Binary dataframe should be inferred as presence_absence
        y_type = infer_y_type(self.y_binary)
        self.assertEqual(y_type, "presence_absence")

        # Binary torch tensor should also work
        y_tensor = torch.tensor(self.y_binary.values)
        self.assertEqual(infer_y_type(y_tensor), "presence_absence")

    def test_infer_y_type_count(self):
        # Count dataframe should be inferred as count
        y_type = infer_y_type(self.y_count)
        self.assertEqual(y_type, "count")

    def test_infer_y_type_invalid(self):
        # Negative values should raise ValueError
        y_neg = self.y_count.copy()
        y_neg.iloc[0, 0] = -1
        with self.assertRaises(ValueError):
            infer_y_type(y_neg)

        # NaN values should raise ValueError
        y_nan = self.y_binary.copy().astype(float)
        y_nan.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            infer_y_type(y_nan)

    def test_prepare_data(self):
        data = prepare_data(self.x, self.y_binary, cuda=False)

        # Check standardisation of x (uses ddof=1 by default in pandas)
        x_np = data["x"].cpu().numpy()
        np.testing.assert_array_almost_equal(
            x_np.mean(axis=0), np.zeros(self.n_env), decimal=5
        )
        np.testing.assert_array_almost_equal(
            x_np.std(axis=0, ddof=1), np.ones(self.n_env), decimal=5
        )

        # Check shapes and types
        self.assertEqual(data["x"].shape, (self.n_sites, self.n_env))
        self.assertEqual(data["y"].shape, (self.n_sites, self.n_species))
        self.assertEqual(data["x"].dtype, torch.float32)
        self.assertEqual(data["y"].dtype, torch.float32)
        self.assertEqual(list(data["y_columns"]), list(self.y_binary.columns))
        self.assertEqual(list(data["site_index"]), list(self.y_binary.index))

    def test_mcmc_batch_size_constraint(self):
        # Should raise ValueError when batch_size is provided with method='mcmc'
        with self.assertRaises(ValueError):
            compute_dark_diversity(
                y=self.y_binary,
                x=self.x,
                model_type="linear",
                method="mcmc",
                batch_size=10,
            )

    def test_mcmc_unexpected_kwargs(self):
        # MCMC should run and ignore SVI-specific keyword arguments without raising TypeError
        # We run with very few samples and warmup steps so it runs quickly
        model = PMFDark(
            model_type="linear",
            num_factors=1,
            method="mcmc",
        )
        model.fit(
            y=self.y_binary,
            x=self.x,
            num_samples=2,
            warmup_steps=2,
            num_iterations=2500,  # SVI parameter, should be ignored
            lr=0.01,              # SVI parameter, should be ignored
        )
        self.assertTrue(model.is_fitted)
        self.assertEqual(model.distribution().shape, self.y_binary.shape)

    def test_svi_predictions_shape_and_chunking(self):
        # Fix seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Run full-batch prediction
        pred_full = compute_dark_diversity(
            y=self.y_binary,
            x=self.x,
            model_type="linear",
            num_factors=1,
            method="svi",
            num_iterations=20,
            return_means=True,
            pred_batch_size=None,
        )

        # Reset seed to get identical fit
        torch.manual_seed(42)
        np.random.seed(42)

        # Run chunked prediction (batch size 5)
        pred_chunk = compute_dark_diversity(
            y=self.y_binary,
            x=self.x,
            model_type="linear",
            num_factors=1,
            method="svi",
            num_iterations=20,
            return_means=True,
            pred_batch_size=5,
        )

        # Compare outcomes (allow small floating-point differences)
        pd.testing.assert_frame_equal(
            pred_full, pred_chunk, check_exact=False, rtol=1e-5, atol=1e-6
        )
        self.assertEqual(pred_full.shape, self.y_binary.shape)

    def test_infer_y_type_categorical(self):
        # Dataframe with categorical (non-numeric) input
        y_cat = pd.DataFrame(
            [["A", "B"], ["B", "A"]],
            index=["site_0", "site_1"],
            columns=["spec_0", "spec_1"],
        )
        # Should raise ValueError as categorical is not supported yet
        with self.assertRaises(ValueError):
            infer_y_type(y_cat)

    def test_prepare_data_categorical_auto(self):
        # Create a df with categorical types: category, object, bool
        x_cat = self.x.copy()
        x_cat["cat_col"] = ["A", "B"] * 10
        x_cat["bool_col"] = [True, False] * 10

        data = prepare_data(x_cat, self.y_binary, cuda=False)
        x_np = data["x"].cpu().numpy()

        # Original 2 float columns + 'cat_col' (A, B -> 2 dummy cols) + 'bool_col' (False, True -> 2 dummy cols)
        # So we should have 2 (continuous) + 2 (dummies for cat_col) + 2 (dummies for bool_col) = 6 columns
        self.assertEqual(data["x"].shape, (self.n_sites, 6))
        self.assertEqual(data["x"].dtype, torch.float32)

        # Continuous columns should be standardized, dummy columns should NOT be standardized (should remain 0.0 or 1.0)
        # Dummy columns are located at the end: columns 2, 3, 4, 5
        dummy_vals = x_np[:, 2:]
        self.assertTrue(np.all(np.isin(dummy_vals, [0.0, 1.0])))

    def test_prepare_data_categorical_explicit(self):
        # Create a df with a label encoded column (integer)
        x_cat = self.x.copy()
        x_cat["landuse"] = [0, 1, 2, 1] * 5  # integer, but we specify it as categorical

        # Run preparation specifying landuse as categorical
        data = prepare_data(
            x_cat, self.y_binary, categorical_cols=["landuse"], cuda=False
        )
        x_np = data["x"].cpu().numpy()

        # 2 float columns + 3 dummy columns for landuse (0, 1, 2) = 5 columns
        self.assertEqual(data["x"].shape, (self.n_sites, 5))

        # Verify dummy columns (columns 2, 3, 4) contain only 0 and 1
        dummy_vals = x_np[:, 2:]
        self.assertTrue(np.all(np.isin(dummy_vals, [0.0, 1.0])))

    def test_prepare_data_categorical_missing(self):
        # Specifying a column not in x should raise ValueError
        with self.assertRaises(ValueError):
            prepare_data(
                self.x, self.y_binary, categorical_cols=["non_existent"], cuda=False
            )

    def test_compute_dark_diversity_with_categorical(self):
        # Smoke test to fit models with categorical environmental variables
        x_cat = self.x.copy()
        x_cat["landuse"] = [0, 1, 2, 1] * 5

        pred = compute_dark_diversity(
            y=self.y_binary,
            x=x_cat,
            model_type="linear",
            num_factors=1,
            method="svi",
            num_iterations=20,
            return_means=True,
            categorical_cols=["landuse"],
        )

        # Shape should match sites x species
        self.assertEqual(pred.shape, (self.n_sites, self.n_species))

    def test_pmf_dark_oo_api(self):
        # Instantiate model
        model = PMFDark(model_type="linear", num_factors=1, method="svi")

        # Fit once
        model.fit(y=self.y_binary, x=self.x, num_iterations=20)
        self.assertTrue(model.is_fitted)

        # Test Return Means = True (DataFrames)
        dist_df = model.distribution(return_means=True)
        pool_df = model.pool(return_means=True)
        dark_df = model.dark(return_means=True)

        self.assertIsInstance(dist_df, pd.DataFrame)
        self.assertIsInstance(pool_df, pd.DataFrame)
        self.assertIsInstance(dark_df, pd.DataFrame)
        self.assertEqual(dist_df.shape, self.y_binary.shape)
        self.assertEqual(pool_df.shape, self.y_binary.shape)
        self.assertEqual(dark_df.shape, self.y_binary.shape)

        # Verify conditional probability calculation on DataFrame:
        # P(Dark) = P(Pool) * (1.0 - P(Distribution))
        expected_dark_df = pool_df * (1.0 - dist_df)
        pd.testing.assert_frame_equal(
            dark_df,
            expected_dark_df,
            check_exact=False,
            rtol=1e-5,
            atol=1e-6,
        )

        # Test Return Means = False (Numpy Arrays)
        dist_np = model.distribution(return_means=False)
        pool_np = model.pool(return_means=False)
        dark_np = model.dark(return_means=False)

        self.assertIsInstance(dist_np, np.ndarray)
        self.assertIsInstance(pool_np, np.ndarray)
        self.assertIsInstance(dark_np, np.ndarray)
        self.assertEqual(dist_np.shape[1:], self.y_binary.shape)
        self.assertEqual(pool_np.shape[1:], self.y_binary.shape)
        self.assertEqual(dark_np.shape[1:], self.y_binary.shape)

        # Verify conditional probability calculation on Numpy Array:
        # P(Dark) = P(Pool) * (1.0 - P(Distribution))
        expected_dark_np = pool_np * (1.0 - dist_np)
        np.testing.assert_allclose(
            dark_np,
            expected_dark_np,
            rtol=1e-5,
            atol=1e-6,
        )

        # Test num_samples specified in fit
        model_small_samples = PMFDark(model_type="linear", num_factors=1, method="svi")
        model_small_samples.fit(
            y=self.y_binary, x=self.x, num_iterations=10, num_samples=15
        )
        dist_small_np = model_small_samples.distribution(return_means=False)
        self.assertEqual(dist_small_np.shape[0], 15)

    def test_float_arguments_casting(self):
        # Pass float inputs for num_factors, num_iterations, num_samples, hidden_size to test casting logic
        # (similar to what R/reticulate sends by default)
        model = PMFDark(
            model_type="bnn",
            num_factors=1.0,
            method="svi",
            hidden_size=10.0,
        )
        model.fit(
            y=self.y_binary,
            x=self.x,
            num_iterations=2.0,
            num_samples=5.0,
            batch_size=10.0,
        )
        self.assertTrue(model.is_fitted)

        # Test with prediction batch size as float
        pred = model.distribution(pred_batch_size=5.0, return_means=True)
        self.assertEqual(pred.shape, self.y_binary.shape)

    def test_demo_datasets(self):
        from pmf_dark import env, survey

        self.assertIsInstance(env, pd.DataFrame)
        self.assertIsInstance(survey, pd.DataFrame)
        self.assertEqual(env.shape, (225, 4))
        self.assertEqual(survey.shape, (225, 100))
        self.assertEqual(
            list(env.columns), ["temperature", "pH", "elevation", "landuse"]
        )
        self.assertTrue(all(col.startswith("sp") for col in survey.columns))


if __name__ == "__main__":
    unittest.main()
