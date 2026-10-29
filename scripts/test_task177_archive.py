"""Regression from the first real portal rejection, with both actual archives."""
import unittest
from pathlib import Path
from check_task177_archive import check

ROOT = Path(__file__).resolve().parents[1]


class PortalInterfaceRegression(unittest.TestCase):
    def test_real_rejected_archive_is_rejected_locally(self):
        with self.assertRaisesRegex(ValueError, 'Portal requires def'):
            check(ROOT / 'submissions/task177-z1.zip')

    def test_corrected_archive_matches_current_source(self):
        check(ROOT / 'submissions/task177-z1-interfacefix.zip')


if __name__ == '__main__':
    unittest.main()
