"""
Unit tests for train.py functions and workflows
Run with: python -m pytest test_training.py -v
"""

import argparse
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Import functions from train.py
from train import (
    MakeEdgeHist,
    deltaR_calc,
    get_callbacks,
    kT_calc,
    main,
    mass2_calc,
    train_dataGenerator,
    train_loadAllData,
    z_calc,
)


class TestPhysicsCalculations(unittest.TestCase):
    """Test physics calculation functions"""

    def setUp(self):
        """Set up test data"""
        # Sample physics values
        self.eta1 = np.array([0.5, 1.0, -0.5])
        self.phi1 = np.array([0.0, np.pi / 2, np.pi])
        self.eta2 = np.array([1.0, 0.5, 0.0])
        self.phi2 = np.array([np.pi / 4, 0.0, -np.pi / 2])

        self.pt1 = np.array([10.0, 20.0, 30.0])
        self.pt2 = np.array([15.0, 25.0, 5.0])

        # 4-momentum vectors [E, px, py, pz]
        self.p1 = np.array([[100, 10, 5, 20], [200, 15, 10, 30]])
        self.p2 = np.array([[150, 8, 12, 25], [180, 20, 8, 15]])

    def test_deltaR_calc(self):
        """Test deltaR calculation"""
        dR = deltaR_calc(self.eta1, self.phi1, self.eta2, self.phi2)

        # Should return numpy array
        self.assertIsInstance(dR, np.ndarray)
        self.assertEqual(len(dR), 3)

        # All values should be positive
        self.assertTrue(np.all(dR >= 0))

        # Test specific case: same eta/phi should give dR=0
        dR_same = deltaR_calc(
            np.array([1.0]), np.array([0.5]), np.array([1.0]), np.array([0.5])
        )
        self.assertAlmostEqual(dR_same[0], 0.0, places=10)

    def test_deltaR_phi_wrapping(self):
        """Test phi wrapping in deltaR calculation"""
        # Test phi > pi case
        eta1, eta2 = np.array([0.0]), np.array([0.0])
        phi1, phi2 = np.array([3 * np.pi / 2]), np.array([np.pi / 4])
        dR = deltaR_calc(eta1, phi1, eta2, phi2)

        # Should handle phi wrapping correctly
        self.assertTrue(dR[0] < np.pi)  # dR should be reasonable

        # Test phi < -pi case
        phi1, phi2 = np.array([-3 * np.pi / 2]), np.array([np.pi / 4])
        dR = deltaR_calc(eta1, phi1, eta2, phi2)
        self.assertTrue(dR[0] < np.pi)

    def test_kT_calc(self):
        """Test kT calculation"""
        dR = np.array([0.5, 1.0, 2.0])
        kT = kT_calc(self.pt1, self.pt2, dR)

        # Should return numpy array
        self.assertIsInstance(kT, np.ndarray)
        self.assertEqual(len(kT), 3)

        # All values should be positive
        self.assertTrue(np.all(kT >= 0))

        # kT should be min(pt1, pt2) * dR
        expected_kT = np.minimum(self.pt1, self.pt2) * dR
        np.testing.assert_array_almost_equal(kT, expected_kT)

    def test_z_calc(self):
        """Test z calculation"""
        z = z_calc(self.pt1, self.pt2)

        # Should return numpy array
        self.assertIsInstance(z, np.ndarray)
        self.assertEqual(len(z), 3)

        # All values should be between 0 and 1
        self.assertTrue(np.all(z >= 0))
        self.assertTrue(np.all(z <= 1))

        # Test specific case
        pt1, pt2 = np.array([10.0]), np.array([20.0])
        z_test = z_calc(pt1, pt2)
        expected = 10.0 / (10.0 + 20.0)
        self.assertAlmostEqual(z_test[0], expected, places=10)

    def test_mass2_calc(self):
        """Test invariant mass squared calculation"""
        # Reshape to match expected input format
        p1 = self.p1.reshape(2, 1, 4)
        p2 = self.p2.reshape(2, 1, 4)

        m2 = mass2_calc(p1, p2)

        # Should return numpy array
        self.assertIsInstance(m2, np.ndarray)
        self.assertEqual(m2.shape, (2, 1))

        # Mass squared can be positive or negative (for timelike/spacelike)
        # But should be finite
        self.assertTrue(np.all(np.isfinite(m2)))


class TestCallbacks(unittest.TestCase):
    """Test callback creation"""

    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.path_out = self.temp_dir + "/"

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_get_callbacks(self):
        """Test callback creation"""
        sample_size = 1000
        batch_size = 32

        callbacks = get_callbacks(self.path_out, sample_size, batch_size)

        # Should return list of callbacks
        self.assertIsInstance(callbacks, list)
        self.assertEqual(len(callbacks), 6)

        # Check callback types
        callback_types = [type(cb).__name__ for cb in callbacks]
        expected_types = [
            "EarlyStopping",
            "CyclicLR",
            "TerminateOnNaN",
            "CSVLogger",
            "ModelCheckpoint",
            "TensorBoard",
        ]

        for expected_type in expected_types:
            self.assertIn(expected_type, callback_types)


class TestPlotting(unittest.TestCase):
    """Test plotting functions"""

    def setUp(self):
        """Set up temporary directory and test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_plot.png")
        self.edge_feat = np.random.normal(0, 1, 1000)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.hist")
    @patch("matplotlib.pyplot.xlabel")
    @patch("matplotlib.pyplot.ylabel")
    def test_MakeEdgeHist(
        self, mock_ylabel, mock_xlabel, mock_hist, mock_figure, mock_close, mock_savefig
    ):
        """Test edge histogram creation"""

        MakeEdgeHist(
            self.edge_feat,
            xname="Test X",
            outputname=self.output_path,
            nbins=50,
            density=True,
            yname="Test Y",
        )

        # Check that plotting functions were called
        mock_figure.assert_called_once()
        mock_hist.assert_called_once()
        mock_xlabel.assert_called_once_with("Test X")
        mock_ylabel.assert_called_once_with("Test Y")
        mock_savefig.assert_called_once_with(self.output_path)
        mock_close.assert_called_once()


class TestArgumentParsing(unittest.TestCase):
    """Test command line argument parsing"""

    def setUp(self):
        """Set up test arguments"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    @patch("train.train_dataGenerator")
    @patch("os.makedirs")
    def test_main_dataGenerator(self, mock_makedirs, mock_train_func):
        """Test main function with dataGenerator workflow"""

        test_args = [
            "--workflowType",
            "dataGenerator",
            "--input",
            "/fake/input/path",
            "--output",
            self.temp_dir,
            "--mode",
            "0",
            "--epochs",
            "10",
            "--batch-size",
            "64",
            "--normFac",
            "100",
            "--loss-symmetry",
            "--symmetry-weight",
            "0.5",
        ]

        with patch("sys.argv", ["train.py"] + test_args):
            main()

        # Check that output directory creation was attempted
        mock_makedirs.assert_called_once()

        # Check that training function was called
        mock_train_func.assert_called_once()

        # Verify arguments passed to training function
        args = mock_train_func.call_args[0][0]
        self.assertEqual(args.workflowType, "dataGenerator")
        self.assertEqual(args.mode, 0)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.normFac, 100)
        self.assertTrue(args.loss_symmetry)
        self.assertEqual(args.symmetry_weight, 0.5)

    @patch("train.train_loadAllData")
    @patch("os.makedirs")
    def test_main_loadAllData(self, mock_makedirs, mock_train_func):
        """Test main function with loadAllData workflow"""

        test_args = [
            "--workflowType",
            "loadAllData",
            "--input",
            "/fake/input/path",
            "--output",
            self.temp_dir,
            "--mode",
            "1",
            "--units",
            "64",
            "32",
            "16",
        ]

        with patch("sys.argv", ["train.py"] + test_args):
            main()

        mock_makedirs.assert_called_once()
        mock_train_func.assert_called_once()

        args = mock_train_func.call_args[0][0]
        self.assertEqual(args.workflowType, "loadAllData")
        self.assertEqual(args.mode, 1)


class TestTrainingWorkflows(unittest.TestCase):
    """Test training workflow functions (mocked)"""

    def setUp(self):
        """Set up test arguments"""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock arguments
        self.mock_args = argparse.Namespace(
            maxNPF=50,
            normFac=100,
            loss_symmetry=True,
            symmetry_weight=0.5,
            epochs=2,
            batch_size=32,
            mode=0,
            input=self.temp_dir,
            output=self.temp_dir + "/output/",
            quantized=None,
            model="dense_embedding",
            units=[32, 16],
            compute_edge_feat=0,
            edge_features=None,
        )

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    @patch("train.test")
    @patch("train.tqdm")
    @patch("train.get_callbacks")
    @patch("train.dense_embedding")
    @patch("train.DataGenerator")
    @patch("train.glob")
    def test_train_dataGenerator_workflow(
        self, mock_glob, mock_data_gen, mock_model, mock_callbacks, mock_tqdm, mock_test
    ):
        """Test dataGenerator training workflow"""

        # Mock file list
        mock_glob.return_value = [
            f"{self.temp_dir}/file1.root",
            f"{self.temp_dir}/file2.root",
            f"{self.temp_dir}/file3.root",
        ]

        # Mock DataGenerator
        mock_gen_instance = MagicMock()
        mock_gen_instance.__len__.return_value = 10
        mock_gen_instance.__getitem__.return_value = (
            [np.random.rand(32, 50, 6), np.random.rand(32, 50, 6)],
            np.random.rand(32, 2),
        )
        mock_gen_instance.emb_input_dim = {0: 10, 1: 5}
        mock_data_gen.return_value = mock_gen_instance

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.fit.return_value = MagicMock()
        mock_model_instance.predict.return_value = np.random.rand(100, 2)
        mock_model.return_value = mock_model_instance

        # Mock callbacks
        mock_callbacks.return_value = []

        # Mock tqdm iteration
        mock_tqdm.tqdm.return_value = [
            (
                [np.random.rand(32, 50, 6), np.random.rand(32, 50, 6)],
                np.random.rand(32, 2),
            )
        ]

        # Create output directory
        os.makedirs(self.mock_args.output, exist_ok=True)

        # Run training
        train_dataGenerator(self.mock_args)

        # Verify key function calls
        self.assertEqual(mock_data_gen.call_count, 3)  # train, valid, test
        mock_model.assert_called_once()
        mock_model_instance.compile.assert_called_once()
        mock_model_instance.fit.assert_called_once()
        mock_test.assert_called_once()

    @patch("train.test")
    @patch("train.train_test_split")
    @patch("train.read_input")
    @patch("train.preProcessing")
    @patch("train.dense_embedding")
    @patch("train.get_callbacks")
    @patch("train.glob")
    @patch("os.system")
    @patch("os.path.isfile")
    def test_train_loadAllData_workflow(
        self,
        mock_isfile,
        mock_system,
        mock_glob,
        mock_callbacks,
        mock_model,
        mock_preproc,
        mock_read_input,
        mock_split,
        mock_test,
    ):
        """Test loadAllData training workflow"""

        # Mock file operations
        mock_glob.return_value = [f"{self.temp_dir}/file1.root"]

        def mock_isfile_side_effect(path):
            if path.endswith(".h5"):
                return False  # h5 file doesn't exist
            return True  # other files exist

        mock_isfile.side_effect = mock_isfile_side_effect

        # Mock data loading
        mock_read_input.return_value = (
            np.random.rand(1000, 50, 6),  # X
            np.random.rand(1000, 2),  # Y
        )

        # Mock preprocessing
        mock_preproc.return_value = (
            np.random.rand(1000, 50, 6),  # Xi
            np.random.rand(1000, 50, 6),  # Xp
            np.random.randint(0, 10, (1000, 50)),  # Xc1
            np.random.randint(0, 5, (1000, 50)),  # Xc2
        )

        # Mock train/test split
        mock_split.side_effect = [
            (np.arange(700), np.arange(700, 1000)),  # train/test
            (np.arange(600), np.arange(600, 700)),  # train/valid
        ]

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.fit.return_value = MagicMock()
        mock_model_instance.predict.return_value = np.random.rand(300, 2)
        mock_model.return_value = mock_model_instance

        # Mock callbacks
        mock_callbacks.return_value = []

        # Create output directory
        os.makedirs(self.mock_args.output, exist_ok=True)

        # Run training
        train_loadAllData(self.mock_args)

        # Verify key function calls
        mock_system.assert_called_once()  # h5 conversion
        mock_read_input.assert_called_once()
        self.assertEqual(mock_preproc.call_count, 2)
        mock_model.assert_called_once()
        mock_model_instance.compile.assert_called_once()
        mock_model_instance.fit.assert_called_once()
        mock_test.assert_called_once()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    # not currently used
    '''
    def test_deltaR_calc_edge_cases(self):
        """Test deltaR calculation with edge cases"""
        # Empty arrays
        with self.assertRaises(IndexError):
            deltaR_calc(np.array([]), np.array([]), np.array([]), np.array([]))

        # Mismatched array sizes
        with self.assertRaises((ValueError, IndexError)):
            deltaR_calc(
                np.array([1.0]), np.array([1.0]),
                np.array([1.0, 2.0]), np.array([1.0])
            )
    '''

    def test_physics_calc_with_zeros(self):
        """Test physics calculations with zero values"""
        # Zero momentum should handle gracefully
        pt_zero = np.array([0.0, 10.0, 0.0])
        pt_nonzero = np.array([5.0, 15.0, 20.0])

        z = z_calc(pt_zero, pt_nonzero)

        # Should not produce NaN or inf
        self.assertTrue(np.all(np.isfinite(z)))

        # Zero momentum should give z=0
        self.assertAlmostEqual(z[0], 0.0, places=10)
        self.assertAlmostEqual(z[2], 0.0, places=10)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
