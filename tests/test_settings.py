import os
import unittest
from pathlib import Path
from unittest.mock import patch

from settings import REPOSITORY_ROOT, Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_are_repository_relative(self):
        settings = Settings()
        self.assertEqual(settings.repository_root, REPOSITORY_ROOT)
        self.assertEqual(settings.data_dir, REPOSITORY_ROOT / "data")
        self.assertEqual(settings.model_path, REPOSITORY_ROOT / "artifacts" / "models" / "sft")

    def test_relative_environment_path_is_resolved(self):
        with patch.dict(os.environ, {"SECANAI_DATA_DIR": "custom-data"}, clear=False):
            self.assertEqual(Settings().data_dir, REPOSITORY_ROOT / "custom-data")
