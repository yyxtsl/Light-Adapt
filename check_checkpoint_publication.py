"""Exercise publication safeguards using tiny synthetic files and no network."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import publish_checkpoints as publication


class PublicationContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        (self.assets / 'one.pth').write_bytes(b'synthetic checkpoint, not a tensor')
        self.row = {'id': 'one', 'filename': 'one.pth', 'path': 'one.pth',
                    'bytes': (self.assets / 'one.pth').stat().st_size,
                    'sha256': hashlib.sha256((self.assets / 'one.pth').read_bytes()).hexdigest()}
        self.plan = {'source_root': str(self.assets), 'files': [self.row],
                     'default_upload_subset': ['one']}

    def run_plan(self, *extra):
        path = self.root / 'plan.json'
        path.write_text(json.dumps(self.plan), encoding='utf-8')
        with redirect_stdout(io.StringIO()):
            return publication.main(['--plan', str(path), *extra])

    def test_default_verifies_without_credentials_or_network(self):
        with patch.object(publication.getpass, 'getpass', side_effect=AssertionError('prompted')):
            self.assertEqual(self.run_plan(), 0)

    def test_corrupt_weight_is_rejected_before_upload(self):
        (self.assets / 'one.pth').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'size/hash'):
            self.run_plan('--upload')

    def test_path_traversal_is_rejected(self):
        (self.root / 'outside.pth').write_bytes(b'outside')
        self.row['path'] = '../outside.pth'
        with self.assertRaisesRegex(ValueError, 'leaves'):
            self.run_plan()

    def test_unknown_or_duplicate_selected_ids_rejected(self):
        for values in [('missing',), ('one', 'one')]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.run_plan('--ids', *values)

    def test_duplicate_plan_ids_rejected(self):
        self.plan['files'].append(dict(self.row))
        with self.assertRaisesRegex(ValueError, 'Duplicate checkpoint IDs'):
            self.run_plan()

    def test_reserved_remote_filename_rejected(self):
        self.row['filename'] = 'README.md'
        with self.assertRaisesRegex(ValueError, 'reserved'):
            self.run_plan()

    def test_upload_requires_explicit_repository_and_model_card(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            self.run_plan('--upload')


if __name__ == '__main__':
    unittest.main()
