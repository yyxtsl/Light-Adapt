"""CPU contracts for portable controls, DWIE conversion and label isolation."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import reproduce_diagnostics as diagnostics
import reproduce_dwie as dwie
import reproduce_redocred as redocred


class WorkflowContracts(unittest.TestCase):
    def test_help_has_no_runtime_import(self):
        for module in [diagnostics, dwie]:
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(module.main([]), 0)
            self.assertIn('usage:', output.getvalue())
        self.assertNotIn('torch', sys.modules)

    def test_generic_training_never_accepts_a_task_checkpoint(self):
        values = ['--mode', 'validate', '--validate-for', 'train', '--variant', 'G',
                  '--model-dir', '.', '--data-dir', '.', '--checkpoint', 'source.pth']
        args = redocred.normalized_args(redocred.parser().parse_args(values))
        with self.assertRaisesRegex(ValueError, 'Generic'):
            redocred.validate(args)

    def test_all_redocred_control_names_are_explicit(self):
        for name in ['C', 'B', 'C-off', 'D', 'G', 'U']:
            args = redocred.parser().parse_args(['--mode', 'validate', '--variant', name])
            self.assertEqual(args.variant, name)

    def test_dwie_training_cannot_select_on_test(self):
        args = dwie.parser().parse_args(['--mode', 'train', '--variant', 'G', '--split', 'test'])
        with self.assertRaisesRegex(ValueError, 'Dev'):
            dwie.validation(args)

    def test_dwie_benchmark_cannot_read_test(self):
        args = dwie.parser().parse_args(['--mode', 'benchmark', '--split', 'test'])
        with self.assertRaisesRegex(ValueError, 'Dev'):
            dwie.validation(args)

    def annotation(self, tag):
        return {'id': 'synthetic', 'content': 'Alice met Bob.', 'tags': [tag],
                'concepts': [{'concept': 0}, {'concept': 1}, {'concept': 2}],
                'mentions': [{'concept': 0, 'begin': 0, 'end': 5, 'text': 'Alice'},
                             {'concept': 1, 'begin': 10, 'end': 13, 'text': 'Bob'}],
                'relations': [{'s': 0, 'p': 'knows', 'o': 1},
                              {'s': 2, 'p': 'unanchored', 'o': 2}]}

    def test_conversion_preserves_unanchored_self_gold(self):
        annotation = self.annotation('train')
        digest = hashlib.sha1(annotation['content'].encode()).hexdigest()
        with redocred.project_imports():
            record = dwie.convert_annotation(annotation, digest, True)
        self.assertEqual(len(record['entities']), 3)
        self.assertEqual(record['entities'][2]['mentions'], [])
        self.assertIn({'h': 2, 'r': 'unanchored', 't': 2}, record['relations'])

    def test_conversion_rejects_mismatched_article_hash(self):
        with redocred.project_imports():
            with self.assertRaisesRegex(ValueError, 'SHA1'):
                dwie.convert_annotation(self.annotation('train'), '0' * 40, True)

    def test_test_conversion_requires_explicit_test_tags(self):
        annotation = self.annotation('train')
        digest = hashlib.sha1(annotation['content'].encode()).hexdigest()
        with redocred.project_imports():
            with self.assertRaisesRegex(ValueError, 'Test-only'):
                dwie.convert_annotation(annotation, digest, False)

    def test_evaluation_strips_gold_before_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'features').mkdir()
            (root / 'gold').mkdir()
            split = redocred.read_json(dwie.ROOT / 'metadata/dwie/target_split.json')['dev']
            rows = [{'id': ident, 'inputs': {'content_ids': [1], 'entity_ids': [0],
                                          'entity_pos': [[]]},
                     'gold': [{'h': 0, 'r': 'private-label', 't': 0}]} for ident in split]
            (root / 'features/dev.jsonl').write_text(
                '\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            (root / 'gold/dev.json').write_text(json.dumps([
                {'id': ident, 'entities': [], 'relations': []} for ident in split]), encoding='utf-8')
            features, _ = dwie.read_features(root, 'dev')
            self.assertTrue(all(row['gold'] == [] for row in features))
            self.assertEqual(features[0]['inputs'], rows[0]['inputs'])

    def test_latency_entity_bins_match_study(self):
        self.assertEqual([diagnostics.bin_name(n) for n in [15, 16, 20, 21, 25, 26]],
                         ['N<=15', '16<=N<=20', '16<=N<=20',
                          '21<=N<=25', '21<=N<=25', 'N>=26'])


if __name__ == '__main__':
    unittest.main()
