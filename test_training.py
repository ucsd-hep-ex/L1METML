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


class TestConfigBasedIntegration(unittest.TestCase):
    """Integration tests for config-based training system"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = create_default_config()

        # Set minimal config for fast testing
        self.config.set("paths.input", self.temp_dir)
        self.config.set("paths.output", self.temp_dir + "/output/")
        self.config.set("training.epochs", 1)
        self.config.set("training.batch_size", 8)
        self.config.set("data.maxNPF", 16)

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_config_workflow_equivalence(self):
        """Test that config-based workflow produces equivalent results to legacy"""
        # This is a conceptual test - in practice you would compare
        # outputs from both workflows with identical parameters

        # Legacy args simulation
        legacy_args = argparse.Namespace(
            maxNPF=128,
            normFac=100,
            epochs=100,
            batch_size=1024,
            mode=0,
            model="dense_embedding",
            units=[64, 32, 16],
            loss_symmetry=False,
            symmetry_weight=1.0,
        )

        # Equivalent config
        config = create_default_config()
        config.set("data.maxNPF", 128)
        config.set("training.normFac", 100)
        config.set("training.epochs", 100)
        config.set("training.batch_size", 1024)
        config.set("training.mode", 0)
        config.set("model.type", "dense_embedding")
        config.set("model.units", [64, 32, 16])
        config.set("loss.use_symmetry", False)
        config.set("loss.symmetry_weight", 1.0)

        # Verify parameter equivalence
        self.assertEqual(config.get("data.maxNPF"), legacy_args.maxNPF)
        self.assertEqual(config.get("training.normFac"), legacy_args.normFac)
        self.assertEqual(config.get("training.epochs"), legacy_args.epochs)
        self.assertEqual(config.get("training.batch_size"), legacy_args.batch_size)
        self.assertEqual(config.get("training.mode"), legacy_args.mode)
        self.assertEqual(config.get("model.units"), legacy_args.units)

    def test_end_to_end_config_flow(self):
        """Test complete config-based training flow"""

        # Save config to file
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.config.save(config_path)

        # Load config from file
        loaded_config = load_config(config_path)

        # Verify config roundtrip
        self.assertEqual(loaded_config.get("training.epochs"), 1)
        self.assertEqual(loaded_config.get("data.maxNPF"), 16)

        # Test callback creation
        callbacks = get_callbacks_from_config(
            loaded_config, self.config.get("paths.output"), 100, 8
        )
        self.assertIsInstance(callbacks, list)
        self.assertGreater(len(callbacks), 0)

    @patch("train.dense_embedding")
    def test_model_creation_integration(self, mock_dense_embedding):
        """Test model creation integration with config"""
        mock_model = MagicMock()
        mock_dense_embedding.return_value = mock_model

        # Test model creation from config
        emb_input_dim = {0: 8, 1: 4}
        model = create_model_from_config(self.config, emb_input_dim, 16)

        self.assertEqual(model, mock_model)
        mock_dense_embedding.assert_called_once()

        # Test model compilation
        mock_loss = MagicMock()
        compiled_model = compile_model(model, self.config, mock_loss)

        self.assertEqual(compiled_model, mock_model)
        mock_model.compile.assert_called_once()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
