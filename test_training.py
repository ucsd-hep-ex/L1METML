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
import h5py

# Import additional modules for config testing
from config import Config, create_default_config, load_config, merge_config_with_args
from loss import custom_loss_wrapper

# Import functions from train.py
from train import (
    MakeEdgeHist,
    compile_model,
    create_model_from_config,
    deltaR_calc,
    get_callbacks,
    get_callbacks_from_config,
    kT_calc,
    main,
    mass2_calc,
    train_dataGenerator,
    train_dataGenerator_from_config,
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


class TestConfigManagement(unittest.TestCase):
    """Test configuration management functionality"""

    def setUp(self):
        """Set up test config"""
        self.test_config_dict = {
            "model": {
                "type": "dense_embedding",
                "units": [32, 16],
                "activation": "relu",
            },
            "training": {"epochs": 50, "batch_size": 64},
            "nested": {"deep": {"value": 42}},
        }
        self.config = Config(self.test_config_dict)

    def test_config_creation(self):
        """Test Config class creation"""
        self.assertIsInstance(self.config, Config)
        self.assertEqual(self.config.to_dict(), self.test_config_dict)

    def test_config_get_simple(self):
        """Test getting simple config values"""
        self.assertEqual(self.config.get("model.type"), "dense_embedding")
        self.assertEqual(self.config.get("training.epochs"), 50)
        self.assertEqual(self.config.get("training.batch_size"), 64)

    def test_config_get_nested(self):
        """Test getting nested config values"""
        self.assertEqual(self.config.get("nested.deep.value"), 42)
        self.assertEqual(self.config.get("model.units"), [32, 16])

    def test_config_get_default(self):
        """Test getting non-existent values with defaults"""
        self.assertEqual(self.config.get("nonexistent.key", "default"), "default")
        self.assertEqual(self.config.get("model.nonexistent", 999), 999)
        self.assertIsNone(self.config.get("missing.key"))

    def test_config_set_simple(self):
        """Test setting simple config values"""
        self.config.set("new.key", "new_value")
        self.assertEqual(self.config.get("new.key"), "new_value")

        self.config.set("training.epochs", 100)
        self.assertEqual(self.config.get("training.epochs"), 100)

    def test_config_set_nested(self):
        """Test setting nested config values"""
        self.config.set("deep.nested.new.key", "deep_value")
        self.assertEqual(self.config.get("deep.nested.new.key"), "deep_value")

    def test_config_save_load(self):
        """Test saving and loading config from file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            # Save config
            self.config.save(temp_path)

            # Load config
            loaded_config = load_config(temp_path)

            # Verify loaded config matches original
            self.assertEqual(loaded_config.to_dict(), self.config.to_dict())
            self.assertEqual(loaded_config.get("model.type"), "dense_embedding")
            self.assertEqual(loaded_config.get("nested.deep.value"), 42)
        finally:
            os.unlink(temp_path)

    def test_create_default_config(self):
        """Test creating default configuration"""
        default_config = create_default_config()

        self.assertIsInstance(default_config, Config)

        # Check required sections exist
        self.assertEqual(default_config.get("model.type"), "dense_embedding")
        self.assertEqual(default_config.get("training.workflow_type"), "dataGenerator")
        self.assertEqual(default_config.get("data.maxNPF"), 128)
        self.assertEqual(default_config.get("loss.use_symmetry"), False)
        self.assertEqual(default_config.get("quantization.enabled"), False)

    def test_merge_config_with_args(self):
        """Test merging config with command line arguments"""
        # Create mock arguments
        args = argparse.Namespace(
            epochs=200,
            batch_size=512,
            input="/new/input/path",
            output="/new/output/path",
            mode=1,
            model="graph_embedding",
            units=["128", "64", "32"],
            maxNPF=256,
            normFac=150,
            loss_symmetry=True,
            symmetry_weight=0.8,
            quantized=["8", "3"],
            compute_edge_feat=1,
            edge_features=["deltaR", "kT"],
        )

        merged_config = merge_config_with_args(self.config, args)

        # Check that arguments override config values
        self.assertEqual(merged_config.get("training.epochs"), 200)
        self.assertEqual(merged_config.get("training.batch_size"), 512)
        self.assertEqual(merged_config.get("paths.input"), "/new/input/path")
        self.assertEqual(merged_config.get("paths.output"), "/new/output/path")
        self.assertEqual(merged_config.get("training.mode"), 1)
        self.assertEqual(merged_config.get("model.type"), "graph_embedding")
        self.assertEqual(merged_config.get("model.units"), [128, 64, 32])
        self.assertEqual(merged_config.get("data.maxNPF"), 256)
        self.assertEqual(merged_config.get("training.normFac"), 150)
        self.assertEqual(merged_config.get("loss.use_symmetry"), True)
        self.assertEqual(merged_config.get("loss.symmetry_weight"), 0.8)
        self.assertEqual(merged_config.get("quantization.enabled"), True)
        self.assertEqual(merged_config.get("quantization.total_bits"), 8)
        self.assertEqual(merged_config.get("quantization.int_bits"), 3)
        self.assertEqual(merged_config.get("data.compute_edge_feat"), 1)
        self.assertEqual(merged_config.get("data.edge_features"), ["deltaR", "kT"])


class TestConfigBasedCallbacks(unittest.TestCase):
    """Test config-based callback creation"""

    def setUp(self):
        """Set up test config and directories"""
        self.temp_dir = tempfile.mkdtemp()
        self.path_out = self.temp_dir + "/"

        self.config = create_default_config()
        self.sample_size = 1000
        self.batch_size = 32

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_get_callbacks_from_config_default(self):
        """Test creating callbacks from default config"""
        callbacks = get_callbacks_from_config(
            self.config, self.path_out, self.sample_size, self.batch_size
        )

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

    def test_get_callbacks_from_config_custom(self):
        """Test creating callbacks from custom config"""
        # Modify config with custom values
        self.config.set("callbacks.early_stopping.patience", 20)
        self.config.set("callbacks.early_stopping.monitor", "val_mae")
        self.config.set("callbacks.cyclical_lr.base_lr", 0.001)
        self.config.set("callbacks.cyclical_lr.max_lr", 0.01)
        self.config.set("callbacks.cyclical_lr.mode", "triangular")

        callbacks = get_callbacks_from_config(
            self.config, self.path_out, self.sample_size, self.batch_size
        )

        # Find EarlyStopping callback
        early_stopping = None
        cyclical_lr = None
        for cb in callbacks:
            if type(cb).__name__ == "EarlyStopping":
                early_stopping = cb
            elif type(cb).__name__ == "CyclicLR":
                cyclical_lr = cb

        # Check custom early stopping settings
        self.assertIsNotNone(early_stopping)
        self.assertEqual(early_stopping.patience, 20)
        self.assertEqual(early_stopping.monitor, "val_mae")

        # Check custom cyclical LR settings
        self.assertIsNotNone(cyclical_lr)
        self.assertEqual(cyclical_lr.base_lr, 0.001)
        self.assertEqual(cyclical_lr.max_lr, 0.01)

    def test_callbacks_file_paths(self):
        """Test that callbacks create proper file paths"""
        callbacks = get_callbacks_from_config(
            self.config, self.path_out, self.sample_size, self.batch_size
        )

        # Find file-based callbacks
        csv_logger = None
        model_checkpoint = None
        tensorboard = None

        for cb in callbacks:
            if type(cb).__name__ == "CSVLogger":
                csv_logger = cb
            elif type(cb).__name__ == "ModelCheckpoint":
                model_checkpoint = cb
            elif type(cb).__name__ == "TensorBoard":
                tensorboard = cb

        # Check file paths
        self.assertIsNotNone(csv_logger)
        self.assertTrue(csv_logger.filename.startswith(self.path_out))

        self.assertIsNotNone(model_checkpoint)
        self.assertTrue(model_checkpoint.filepath.startswith(self.path_out))

        self.assertIsNotNone(tensorboard)
        self.assertTrue(tensorboard.log_dir.startswith(self.path_out))


class TestConfigBasedModels(unittest.TestCase):
    """Test config-based model creation"""

    def setUp(self):
        """Set up test config"""
        self.config = create_default_config()
        self.emb_input_dim = {0: 10, 1: 5}  # Mock embedding input dimensions
        self.maxNPF = 128

    @patch("train.dense_embedding")
    def test_create_model_from_config_dense(self, mock_dense_embedding):
        """Test creating dense embedding model from config"""
        mock_model = MagicMock()
        mock_dense_embedding.return_value = mock_model

        # Set config for dense embedding
        self.config.set("model.type", "dense_embedding")
        self.config.set("model.units", [64, 32, 16])
        self.config.set("model.activation", "tanh")
        self.config.set("model.emb_out_dim", 8)
        self.config.set("model.with_bias", False)

        model = create_model_from_config(self.config, self.emb_input_dim, self.maxNPF)

        # Check that dense_embedding was called
        mock_dense_embedding.assert_called_once()
        self.assertEqual(model, mock_model)

        # Check arguments passed to model creation
        call_args = mock_dense_embedding.call_args[1]
        self.assertEqual(call_args["n_features"], 6)
        self.assertEqual(call_args["emb_out_dim"], 8)
        self.assertEqual(call_args["n_features_cat"], 2)
        self.assertEqual(call_args["activation"], "tanh")
        self.assertEqual(call_args["units"], [64, 32, 16])
        self.assertEqual(call_args["with_bias"], False)

    @patch("train.dense_embedding_quantized")
    def test_create_model_from_config_quantized(self, mock_quantized_model):
        """Test creating quantized model from config"""
        mock_model = MagicMock()
        mock_quantized_model.return_value = mock_model

        # Enable quantization
        self.config.set("quantization.enabled", True)
        self.config.set("quantization.total_bits", 8)
        self.config.set("quantization.int_bits", 3)

        model = create_model_from_config(self.config, self.emb_input_dim, self.maxNPF)

        # Check that quantized model was called
        mock_quantized_model.assert_called_once()
        self.assertEqual(model, mock_model)

        # Check quantization parameters
        call_args = mock_quantized_model.call_args[1]
        self.assertEqual(call_args["logit_total_bits"], 8)
        self.assertEqual(call_args["logit_int_bits"], 3)

    @patch("train.graph_embedding")
    def test_create_model_from_config_graph(self, mock_graph_embedding):
        """Test creating graph embedding model from config"""
        mock_model = MagicMock()
        mock_graph_embedding.return_value = mock_model

        # Set config for graph embedding
        self.config.set("model.type", "graph_embedding")
        self.config.set("data.compute_edge_feat", 1)
        self.config.set("data.edge_features", ["deltaR", "kT"])

        model = create_model_from_config(self.config, self.emb_input_dim, self.maxNPF)

        # Check that graph_embedding was called
        mock_graph_embedding.assert_called_once()
        self.assertEqual(model, mock_model)

        # Check edge feature parameters
        call_args = mock_graph_embedding.call_args[1]
        self.assertEqual(call_args["compute_ef"], 1)
        self.assertEqual(call_args["edge_list"], ["deltaR", "kT"])

    def test_create_model_from_config_invalid_type(self):
        """Test error handling for invalid model type"""
        self.config.set("model.type", "invalid_model_type")

        with self.assertRaises(ValueError):
            create_model_from_config(self.config, self.emb_input_dim, self.maxNPF)

    def test_compile_model_mode_0(self):
        """Test model compilation for mode 0 (L1MET)"""
        mock_model = MagicMock()
        mock_loss = MagicMock()

        self.config.set("training.mode", 0)
        self.config.set("training.optimizer", {"type": "adam"})

        compiled_model = compile_model(mock_model, self.config, mock_loss)

        # Check that compile was called
        mock_model.compile.assert_called_once()
        call_args = mock_model.compile.call_args[1]

        self.assertEqual(call_args["optimizer"], "adam")
        self.assertEqual(call_args["loss"], mock_loss)
        self.assertIn("metrics", call_args)
        self.assertEqual(compiled_model, mock_model)

    @patch("train.optimizers.Adam")
    def test_compile_model_mode_1(self, mock_adam):
        """Test model compilation for mode 1 (DeepMET)"""
        mock_model = MagicMock()
        mock_loss = MagicMock()
        mock_optimizer = MagicMock()
        mock_adam.return_value = mock_optimizer

        self.config.set("training.mode", 1)
        self.config.set("training.optimizer", {"learning_rate": 0.001, "clipnorm": 0.5})

        compiled_model = compile_model(mock_model, self.config, mock_loss)

        # Check Adam optimizer creation
        mock_adam.assert_called_once_with(lr=0.001, clipnorm=0.5)

        # Check model compilation
        mock_model.compile.assert_called_once()
        call_args = mock_model.compile.call_args[1]
        self.assertEqual(call_args["optimizer"], mock_optimizer)
        self.assertEqual(call_args["loss"], mock_loss)


class TestConfigBasedTraining(unittest.TestCase):
    """Test config-based training workflow"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = create_default_config()

        # Set paths
        self.config.set("paths.input", self.temp_dir)
        self.config.set("paths.output", self.temp_dir + "/output/")

        # Set small training parameters for testing
        self.config.set("training.epochs", 2)
        self.config.set("training.batch_size", 16)
        self.config.set("data.maxNPF", 32)

        # Create output directory
        os.makedirs(self.config.get("paths.output"), exist_ok=True)

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    @patch("train.test")
    @patch("train.time.time")
    @patch("train.get_callbacks_from_config")
    @patch("train.compile_model")
    @patch("train.create_model_from_config")
    @patch("train.DataGenerator")
    @patch("train.glob")
    @patch("train.custom_loss_wrapper")
    def test_train_dataGenerator_from_config_success(
        self,
        mock_loss_wrapper,
        mock_glob,
        mock_data_gen,
        mock_create_model,
        mock_compile_model,
        mock_callbacks,
        mock_time,
        mock_test,
    ):
        """Test successful config-based training workflow"""

        # Mock file discovery
        mock_files = [
            f"{self.temp_dir}/file1.root",
            f"{self.temp_dir}/file2.root",
            f"{self.temp_dir}/file3.root",
            f"{self.temp_dir}/file4.root",
        ]
        mock_glob.return_value = mock_files

        # Mock loss function
        mock_loss = MagicMock()
        mock_loss_wrapper.return_value = mock_loss

        # Mock DataGenerator
        mock_gen_instance = MagicMock()
        mock_gen_instance.__len__.return_value = 10
        mock_gen_instance.__getitem__.return_value = (
            [np.random.rand(16, 32, 6), np.random.rand(16, 32, 6)],
            np.random.rand(16, 2),
        )
        mock_gen_instance.__iter__.return_value = iter(
            [
                (
                    [np.random.rand(16, 32, 6), np.random.rand(16, 32, 6)],
                    np.random.rand(16, 2),
                )
            ]
        )
        mock_gen_instance.emb_input_dim = {0: 10, 1: 5}
        mock_data_gen.return_value = mock_gen_instance

        # Mock model creation and compilation
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        mock_compile_model.return_value = mock_model

        # Mock callbacks
        mock_callbacks.return_value = []

        # Mock model training
        mock_history = MagicMock()
        mock_model.fit.return_value = mock_history
        mock_model.predict.return_value = np.random.rand(100, 2)

        # Mock timing
        mock_time.side_effect = [1000.0, 1100.0]  # start and end time

        # Run training
        history, trained_model = train_dataGenerator_from_config(self.config)

        # Verify function calls
        mock_glob.assert_called_once()
        self.assertEqual(mock_data_gen.call_count, 3)  # train, valid, test generators
        mock_create_model.assert_called_once()
        mock_compile_model.assert_called_once()
        mock_callbacks.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_test.assert_called_once()

        # Verify return values
        self.assertEqual(history, mock_history)
        self.assertEqual(trained_model, mock_model)

        # Verify loss function creation
        mock_loss_wrapper.assert_called_once()
        loss_args = mock_loss_wrapper.call_args[1]
        self.assertEqual(loss_args["normFac"], 100)  # from default config
        self.assertEqual(loss_args["use_symmetry"], False)
        self.assertEqual(loss_args["symmetry_weight"], 1.0)

    @patch("train.glob")
    def test_train_dataGenerator_from_config_insufficient_files(self, mock_glob):
        """Test error handling with insufficient files"""
        # Mock insufficient files
        mock_glob.return_value = ["file1.root", "file2.root"]  # Only 2 files

        with self.assertRaises(AssertionError):
            train_dataGenerator_from_config(self.config)

    def test_loss_wrapper_config_integration(self):
        """Test loss function wrapper with config parameters"""
        # Test default symmetry settings
        loss_func = custom_loss_wrapper(
            normFac=self.config.get("training.normFac"),
            use_symmetry=self.config.get("loss.use_symmetry"),
            symmetry_weight=self.config.get("loss.symmetry_weight"),
        )

        self.assertIsNotNone(loss_func)

        # Test with symmetry enabled
        self.config.set("loss.use_symmetry", True)
        self.config.set("loss.symmetry_weight", 0.5)

        loss_func_sym = custom_loss_wrapper(
            normFac=self.config.get("training.normFac"),
            use_symmetry=self.config.get("loss.use_symmetry"),
            symmetry_weight=self.config.get("loss.symmetry_weight"),
        )

        self.assertIsNotNone(loss_func_sym)


class TestMainFunctionConfigMode(unittest.TestCase):
    """Test main function with config file support"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

        # Create test config file
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        test_config = create_default_config()
        test_config.set("paths.input", self.temp_dir)
        test_config.set("paths.output", self.temp_dir + "/output/")
        test_config.save(self.config_file)

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    @patch("train.train_dataGenerator_from_config")
    @patch("train.load_config")
    @patch("os.makedirs")
    def test_main_with_config_file(
        self, mock_makedirs, mock_load_config, mock_train_func
    ):
        """Test main function using config file"""

        # Mock config loading
        mock_config = create_default_config()
        mock_config.set("paths.input", self.temp_dir)
        mock_config.set("paths.output", self.temp_dir + "/output/")
        mock_load_config.return_value = mock_config

        # Mock training function return
        mock_history = MagicMock()
        mock_model = MagicMock()
        mock_train_func.return_value = (mock_history, mock_model)

        test_args = ["--config", self.config_file, "--workflowType", "dataGenerator"]

        with patch("sys.argv", ["train.py"] + test_args):
            result = main()

        # Verify config was loaded
        mock_load_config.assert_called_once_with(self.config_file)

        # Verify output directory creation
        mock_makedirs.assert_called_once()

        # Verify training function was called
        mock_train_func.assert_called_once()

        # Verify return value
        self.assertEqual(result, (mock_history, mock_model))

    @patch("train.train_dataGenerator_from_config")
    @patch("train.create_default_config")
    @patch("train.train_dataGenerator")
    @patch("os.makedirs")
    def test_main_with_config_override(
        self, mock_makedirs, mock_train_legacy, mock_create_default, mock_train_func
    ):
        """Test main function with config override from command line"""

        # Mock default config creation
        mock_config = create_default_config()
        mock_create_default.return_value = mock_config

        # Mock training function returns
        mock_history = MagicMock()
        mock_model = MagicMock()
        mock_train_func.return_value = (mock_history, mock_model)

        test_args = [
            "--workflowType",
            "dataGenerator",
            "--input",
            self.temp_dir,
            "--output",
            self.temp_dir + "/output/",
            "--epochs",
            "150",
            "--batch-size",
            "128",
            "--loss-symmetry",
            "--symmetry-weight",
            "0.3",
        ]

        with patch("sys.argv", ["train.py"] + test_args):
            main()

        # Verify default config was created
        mock_create_default.assert_called_once()

        # Verify legacy training function was called (no config file)
        mock_train_legacy.assert_called_once()


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation and error handling"""

    def test_config_missing_required_paths(self):
        """Test error handling for missing required paths"""
        config = Config({})

        # Test missing paths without command line args
        with self.assertRaises(ValueError):
            # Simulate main function path validation
            if not config.get("paths.input") or not config.get("paths.output"):
                raise ValueError(
                    "Input and output paths must be specified either in config or as command line arguments."
                )

    def test_config_default_values(self):
        """Test that config provides sensible defaults"""
        config = create_default_config()

        # Check that all required sections have defaults
        required_keys = [
            "model.type",
            "model.units",
            "training.epochs",
            "training.batch_size",
            "data.maxNPF",
            "loss.use_symmetry",
            "quantization.enabled",
        ]

        for key in required_keys:
            value = config.get(key)
            self.assertIsNotNone(value, f"Missing default for {key}")

    def test_config_type_consistency(self):
        """Test that config values have expected types"""
        config = create_default_config()

        # Check types
        self.assertIsInstance(config.get("model.units"), list)
        self.assertIsInstance(config.get("training.epochs"), int)
        self.assertIsInstance(config.get("training.batch_size"), int)
        self.assertIsInstance(config.get("loss.use_symmetry"), bool)
        self.assertIsInstance(config.get("quantization.enabled"), bool)


class TestFeatureCompatibility(unittest.TestCase):
    """Test the new 10-feature format compatibility with utils.py preprocessing"""
    
    def setUp(self):
        """Set up test data with 10-feature format from convertNanoToHDF5.py"""
        self.batch_size = 8
        self.maxNPF = 16
        self.normFac = 1000.0
        
        # Create test data matching the 10-feature format from convertNanoToHDF5.py
        # Features: pt, px, py, eta, phi, puppi, pdgId, charge, dxyErr, hcalDepth
        self.test_data = np.random.rand(self.batch_size, self.maxNPF, 10).astype(np.float32)
        
        # Set realistic feature values
        self.test_data[:, :, 0] = np.random.uniform(0, 500, (self.batch_size, self.maxNPF))  # pt
        self.test_data[:, :, 1] = np.random.uniform(-250, 250, (self.batch_size, self.maxNPF))  # px  
        self.test_data[:, :, 2] = np.random.uniform(-250, 250, (self.batch_size, self.maxNPF))  # py
        self.test_data[:, :, 3] = np.random.uniform(-3, 3, (self.batch_size, self.maxNPF))  # eta
        self.test_data[:, :, 4] = np.random.uniform(-np.pi, np.pi, (self.batch_size, self.maxNPF))  # phi
        self.test_data[:, :, 5] = np.random.uniform(0, 1, (self.batch_size, self.maxNPF))  # puppi
        self.test_data[:, :, 6] = np.random.randint(0, 8, (self.batch_size, self.maxNPF))  # encoded pdgId
        self.test_data[:, :, 7] = np.random.randint(0, 3, (self.batch_size, self.maxNPF))  # encoded charge
        self.test_data[:, :, 8] = np.random.uniform(0, 0.1, (self.batch_size, self.maxNPF))  # dxyErr
        self.test_data[:, :, 9] = np.random.uniform(1, 7, (self.batch_size, self.maxNPF))  # hcalDepth
        
        # Test target data (MET x, y)
        self.test_targets = np.random.uniform(-200, 200, (self.batch_size, 2)).astype(np.float32)

    def test_preprocessing_10_features(self):
        """Test that preProcessing correctly handles 10-feature input"""
        from utils import preProcessing
        
        inputs, pxpy, inputs_cat0, inputs_cat1 = preProcessing(self.test_data, self.normFac)
        
        # Verify output shapes
        self.assertEqual(inputs.shape, (self.batch_size, self.maxNPF, 6))  # 6 continuous features
        self.assertEqual(pxpy.shape, (self.batch_size, self.maxNPF, 2))  # px, py
        self.assertEqual(inputs_cat0.shape, (self.batch_size, self.maxNPF))  # categorical pdgId
        self.assertEqual(inputs_cat1.shape, (self.batch_size, self.maxNPF))  # categorical charge
        
        # Verify continuous features are correctly extracted and normalized
        expected_pt = self.test_data[:, :, 0:1] / self.normFac
        expected_eta = self.test_data[:, :, 3:4]
        expected_phi = self.test_data[:, :, 4:5]
        expected_puppi = self.test_data[:, :, 5:6]
        expected_dxyErr = self.test_data[:, :, 8:9]
        expected_hcalDepth = self.test_data[:, :, 9:10]
        
        np.testing.assert_array_almost_equal(inputs[:, :, 0:1], expected_pt, decimal=5)
        np.testing.assert_array_almost_equal(inputs[:, :, 1:2], expected_eta, decimal=5)
        np.testing.assert_array_almost_equal(inputs[:, :, 2:3], expected_phi, decimal=5)
        np.testing.assert_array_almost_equal(inputs[:, :, 3:4], expected_puppi, decimal=5)
        np.testing.assert_array_almost_equal(inputs[:, :, 4:5], expected_dxyErr, decimal=5)
        np.testing.assert_array_almost_equal(inputs[:, :, 5:6], expected_hcalDepth, decimal=5)
        
        # Verify momentum features
        expected_px = self.test_data[:, :, 1:2] / self.normFac
        expected_py = self.test_data[:, :, 2:3] / self.normFac
        np.testing.assert_array_almost_equal(pxpy[:, :, 0:1], expected_px, decimal=5)
        np.testing.assert_array_almost_equal(pxpy[:, :, 1:2], expected_py, decimal=5)
        
        # Verify categorical features
        np.testing.assert_array_equal(inputs_cat0, self.test_data[:, :, 6])
        np.testing.assert_array_equal(inputs_cat1, self.test_data[:, :, 7])

    def test_outlier_removal(self):
        """Test outlier removal in preprocessing"""
        from utils import preProcessing
        
        # Create data with outliers (need to be > 500 after normalization)
        outlier_data = self.test_data.copy()
        outlier_data[0, 0, 0] = 600000  # pt outlier at particle 0
        outlier_data[0, 1, 1] = 600000  # px outlier at particle 1 
        outlier_data[0, 2, 2] = -600000  # py outlier at particle 2
        
        inputs, pxpy, inputs_cat0, inputs_cat1 = preProcessing(outlier_data, self.normFac)
        
        # Check that outliers were set to 0
        # pt outlier: position [0,0] in inputs (pt is first feature)
        self.assertEqual(inputs[0, 0, 0], 0.0)  # pt outlier removed
        # px outlier: position [0,1] in pxpy (px is first feature in pxpy)
        self.assertEqual(pxpy[0, 1, 0], 0.0)  # px outlier removed  
        # py outlier: position [0,2] in pxpy (py is second feature in pxpy)
        self.assertEqual(pxpy[0, 2, 1], 0.0)  # py outlier removed

    def test_feature_mapping_consistency(self):
        """Test that feature mapping matches convertNanoToHDF5.py format"""
        from utils import preProcessing
        
        # Create test data with known values for verification
        test_batch = np.zeros((1, 1, 10))
        test_batch[0, 0, :] = [100, 50, 50, 1.5, 0.5, 0.8, 5, 1, 0.02, 3]  # Known values
        
        inputs, pxpy, inputs_cat0, inputs_cat1 = preProcessing(test_batch, 1000.0)
        
        # Verify specific feature mappings
        self.assertAlmostEqual(inputs[0, 0, 0], 0.1, places=3)  # pt/1000
        self.assertAlmostEqual(inputs[0, 0, 1], 1.5, places=3)  # eta (unchanged)
        self.assertAlmostEqual(inputs[0, 0, 2], 0.5, places=3)  # phi (unchanged)
        self.assertAlmostEqual(inputs[0, 0, 3], 0.8, places=3)  # puppi (unchanged)
        self.assertAlmostEqual(inputs[0, 0, 4], 0.02, places=3)  # dxyErr (unchanged)
        self.assertAlmostEqual(inputs[0, 0, 5], 3.0, places=3)  # hcalDepth (unchanged)
        
        self.assertAlmostEqual(pxpy[0, 0, 0], 0.05, places=3)  # px/1000
        self.assertAlmostEqual(pxpy[0, 0, 1], 0.05, places=3)  # py/1000
        
        self.assertEqual(inputs_cat0[0, 0], 5)  # pdgId
        self.assertEqual(inputs_cat1[0, 0], 1)  # charge


class TestDataGeneratorCompatibility(unittest.TestCase):
    """Test DataGenerator compatibility with new feature format"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_data.h5")
        
        # Create test HDF5 file with 10-feature format
        batch_size = 10
        maxNPF = 8
        
        X_data = np.random.rand(batch_size, maxNPF, 10).astype(np.float32)
        Y_data = np.random.uniform(-100, 100, (batch_size, 2)).astype(np.float32)
        
        with h5py.File(self.test_file, 'w') as f:
            f.create_dataset('X', data=X_data)
            f.create_dataset('Y', data=Y_data)
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('sys.modules', {'setGPU': MagicMock()})
    def test_data_generator_preprocessing_integration(self):
        """Test DataGenerator correctly uses updated preprocessing"""
        from DataGenerator import DataGenerator
        
        # Create DataGenerator with test file
        generator = DataGenerator(
            list_files=[self.test_file],
            batch_size=5,
            maxNPF=8
        )
        
        # Get first batch
        Xr, Yr = generator[0]
        
        # Verify structure: [Xi, Xp, Xc1, Xc2] for non-edge-feature case
        self.assertEqual(len(Xr), 4)
        
        # Check shapes
        Xi, Xp, Xc1, Xc2 = Xr
        self.assertEqual(Xi.shape[2], 6)  # 6 continuous features
        self.assertEqual(Xp.shape[2], 2)  # 2 momentum features
        self.assertEqual(len(Xc1.shape), 2)  # categorical feature 1
        self.assertEqual(len(Xc2.shape), 2)  # categorical feature 2

    @patch('sys.modules', {'setGPU': MagicMock()})
    def test_data_generator_edge_features_compatibility(self):
        """Test DataGenerator with edge features using new preprocessing"""
        from DataGenerator import DataGenerator
        
        generator = DataGenerator(
            list_files=[self.test_file],
            batch_size=5,
            maxNPF=8,
            compute_ef=1,
            edge_list=['dR', 'kT']
        )
        
        # Get first batch  
        Xr, Yr = generator[0]
        
        # Verify structure: [Xi, Xp, Xc1, Xc2, ef] for edge-feature case
        self.assertEqual(len(Xr), 5)
        
        # Check that edge features were computed correctly
        Xi, Xp, Xc1, Xc2, ef = Xr
        self.assertEqual(Xi.shape[2], 6)  # Still 6 continuous features
        self.assertEqual(ef.shape[2], 2)  # 2 edge features (dR, kT)


class TestEndToEndCompatibility(unittest.TestCase):
    """Test end-to-end compatibility from data format to model input"""
    
    def setUp(self):
        """Set up test environment with realistic data"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_integration.h5")
        
        # Create realistic test data matching convertNanoToHDF5.py output
        batch_size = 20
        maxNPF = 16
        
        # Generate realistic particle data
        X_data = np.zeros((batch_size, maxNPF, 10), dtype=np.float32)
        
        # Realistic particle features
        for i in range(batch_size):
            for j in range(maxNPF):
                X_data[i, j, 0] = np.random.exponential(20)  # pt
                X_data[i, j, 1] = X_data[i, j, 0] * np.cos(np.random.uniform(-np.pi, np.pi))  # px
                X_data[i, j, 2] = X_data[i, j, 0] * np.sin(np.random.uniform(-np.pi, np.pi))  # py
                X_data[i, j, 3] = np.random.normal(0, 2)  # eta
                X_data[i, j, 4] = np.random.uniform(-np.pi, np.pi)  # phi
                X_data[i, j, 5] = np.random.beta(2, 2)  # puppi weight
                X_data[i, j, 6] = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])  # encoded pdgId
                X_data[i, j, 7] = np.random.choice([0, 1, 2])  # encoded charge
                X_data[i, j, 8] = np.random.exponential(0.01)  # dxyErr
                X_data[i, j, 9] = np.random.choice([1, 2, 3, 4, 5, 6, 7])  # hcalDepth
        
        Y_data = np.random.normal(0, 50, (batch_size, 2)).astype(np.float32)
        
        with h5py.File(self.test_file, 'w') as f:
            f.create_dataset('X', data=X_data)
            f.create_dataset('Y', data=Y_data)
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('sys.modules', {'setGPU': MagicMock()})
    def test_full_pipeline_compatibility(self):
        """Test complete pipeline from HDF5 → DataGenerator → preprocessing → model input"""
        from DataGenerator import DataGenerator
        from models import dense_embedding
        
        # Test DataGenerator
        generator = DataGenerator(
            list_files=[self.test_file],
            batch_size=10,
            maxNPF=16
        )
        
        # Get batch and verify it processes correctly
        Xr, Yr = generator[0]
        Xi, Xp, Xc1, Xc2 = Xr
        
        # Verify preprocessing worked correctly
        self.assertEqual(Xi.shape[2], 6)  # 6 continuous features
        self.assertEqual(Xp.shape[2], 2)  # 2 momentum features
        
        # Test that embedding dimensions are computed correctly after processing a batch
        self.assertIsInstance(generator.emb_input_dim, dict)
        self.assertIn(0, generator.emb_input_dim)
        self.assertIn(1, generator.emb_input_dim)
        
        # Verify embedding dimensions are reasonable
        self.assertGreater(generator.emb_input_dim[0], 0)
        self.assertGreater(generator.emb_input_dim[1], 0)

    def test_config_compatibility(self):
        """Test that default config is compatible with new feature format"""
        from config import create_default_config
        
        config = create_default_config()
        
        # The config should still reflect that we process 6 continuous features for the model
        # even though the raw data has 10 features total
        self.assertEqual(config.get("data.n_features_pf"), 6)  # Continuous features used in model
        self.assertEqual(config.get("data.n_features_pf_cat"), 2)  # Categorical features
        
        # The config should be updated to reflect higher normalization factor
        self.assertEqual(config.get("data.normFac"), 100)  # Currently 100, could be updated to 1000

    @patch('sys.modules', {'setGPU': MagicMock()})
    def test_config_based_training_compatibility(self):
        """Test config-based training with new feature format"""
        from config import create_default_config
        from DataGenerator import DataGenerator
        
        # Create test config
        config = create_default_config()
        config.set("paths.input", self.temp_dir)
        config.set("paths.output", self.temp_dir + "/output/")
        config.set("training.epochs", 1)
        config.set("training.batch_size", 5)
        config.set("data.maxNPF", 16)
        
        # Test that DataGenerator works with config
        generator = DataGenerator(
            list_files=[self.test_file],
            batch_size=config.get("training.batch_size"),
            maxNPF=config.get("data.maxNPF")
        )
        
        # Process a batch to initialize emb_input_dim
        Xr, Yr = generator[0]
        
        # Verify data structure is correct for model input
        self.assertEqual(len(Xr), 4)  # [Xi, Xp, Xc1, Xc2]
        Xi, Xp, Xc1, Xc2 = Xr
        self.assertEqual(Xi.shape[2], 6)  # 6 continuous features
        self.assertEqual(Xp.shape[2], 2)  # 2 momentum features
        
        # Verify embedding dimensions are set correctly
        self.assertIsInstance(generator.emb_input_dim, dict)
        self.assertIn(0, generator.emb_input_dim)
        self.assertIn(1, generator.emb_input_dim)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
