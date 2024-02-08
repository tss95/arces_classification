import unittest
import torch
import numpy as np
from global_config import logger, cfg
from src.Scaler_torch import MinMaxScaler, StandardScaler, RobustScaler, LogScaler
import os 
import psutil


class TestMinMaxScaler(unittest.TestCase):
    def setUp(self):
        self.scaler = MinMaxScaler()
        self.data = torch.randn(100, 10000, 3).float() * 100  # Mock data for testing

    def test_fit_global_per_channel(self):
        # Test the fit method with global scaling and per channel
        cfg.scaling.global_or_local = "global"
        cfg.scaling.per_channel = True
        self.scaler.fit(self.data)

        # Check that the maxs and mins are correct per channel
        for i in range(3):  # Assuming there are 3 channels
            self.assertEqual(self.scaler.maxs[i].item(), torch.max(self.data[:,:,i]).item())
            self.assertEqual(self.scaler.mins[i].item(), torch.min(self.data[:,:,i]).item())

    def test_fit_global_not_per_channel(self):
        # Test the fit method with global scaling and not per channel
        cfg.scaling.global_or_local = "global"
        cfg.scaling.per_channel = False
        self.scaler.fit(self.data)

        # Check that the maxs and mins are correct for the whole data
        self.assertEqual(self.scaler.maxs.item(), torch.max(self.data).item())
        self.assertEqual(self.scaler.mins.item(), torch.min(self.data).item())


    def test_transform_global(self):
        # Test the transform method with global scaling
        cfg.scaling.global_or_local = "global"
        self.scaler.fit(self.data)
        transformed = self.scaler.transform(self.data)

        # Check that the transformed data is in the range [0, 1]
        self.assertTrue(torch.all(transformed >= 0) and torch.all(transformed <= 1))

    def test_transform_local(self):
        # Test the transform method with local scaling
        cfg.scaling.global_or_local = "local"
        self.scaler.fit(self.data)
        transformed = self.scaler.transform(self.data)

        # Check that the transformed data is in the range [0, 1]
        self.assertTrue(torch.all(transformed >= 0) and torch.all(transformed <= 1))

    def test_memory_leak(self):
        process = psutil.Process(os.getpid())
        self.scaler.fit(self.data)
        mem_info1 = process.memory_info()
        for _ in range(1000):  # Call transform method 1000 times
            transformed_data = self.scaler.transform(self.data)
        mem_info2 = process.memory_info()
        self.assertAlmostEqual(mem_info1.rss, mem_info2.rss, delta=1e7)  # Allow difference of 10MB


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        self.scaler = StandardScaler()
        self.data = torch.randint(-100, 301, (100, 10000, 3)).float()  # Mock data for testing

    def test_fit_global_per_channel(self):
        # Test the fit method with global scaling and per channel
        cfg.scaling.global_or_local = "global"
        cfg.scaling.per_channel = True
        self.scaler.fit(self.data)

        # Check that the means and stds are correct per channel
        for i in range(3):  # Assuming there are 3 channels
            self.assertAlmostEqual(self.scaler.means[i].item(), torch.mean(self.data[:,:,i]).item(), places=5)
            self.assertAlmostEqual(self.scaler.stds[i].item(), torch.std(self.data[:,:,i]).item(), places=5)

    def test_fit_global_not_per_channel(self):
        # Test the fit method with global scaling and not per channel
        cfg.scaling.global_or_local = "global"
        cfg.scaling.per_channel = False
        self.scaler.fit(self.data)

        # Check that the means and stds are correct for the whole data
        self.assertAlmostEqual(self.scaler.means.item(), torch.mean(self.data).item(), places=3)
        self.assertAlmostEqual(self.scaler.stds.item(), torch.std(self.data).item(), places=3)

    def test_fit_transform_global_per_channel(self):
        # Test the fit and transform methods with global scaling and per channel
        cfg.scaling.global_or_local = "global"
        cfg.scaling.per_channel = True
        self.scaler.fit(self.data)
        transformed_data = self.scaler.transform(self.data)

        # Check that the means and stds of the transformed data are correct per channel
        for i in range(3):  # Assuming there are 3 channels
            self.assertAlmostEqual(torch.mean(transformed_data[:,:,i]).item(), 0, places=4)
            self.assertAlmostEqual(torch.std(transformed_data[:,:,i]).item(), 1, places=4)

    def test_fit_transform_global_not_per_channel(self):
        # Test the fit and transform methods with global scaling and not per channel
        cfg.scaling.global_or_local = "global"
        cfg.scaling.per_channel = False
        self.scaler.fit(self.data)
        transformed_data = self.scaler.transform(self.data)

        # Check that the means and stds of the transformed data are correct for the whole data
        self.assertAlmostEqual(torch.mean(transformed_data).item(), 0, places=4)
        self.assertAlmostEqual(torch.std(transformed_data).item(), 1, places=4)

    def test_memory_leak(self):
        process = psutil.Process(os.getpid())
        cfg.scaling.global_or_local = "global"
        cfg.scaling.per_channel = False
        self.scaler.fit(self.data)
        mem_info1 = process.memory_info()
        for _ in range(1000):  # Call transform method 1000 times
            transformed_data = self.scaler.transform(self.data)
        mem_info2 = process.memory_info()
        self.assertAlmostEqual(mem_info1.rss, mem_info2.rss, delta=1e7)  # Allow difference of 10MB


if __name__ == '__main__':
    unittest.main()