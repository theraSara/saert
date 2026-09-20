from pathlib import Path
import json
import tempfile
import unittest
from src.report_bundle import digest, verify_bundle

class ReportBundleTests(unittest.TestCase):
    def test_valid_bundle_and_modified_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);p=root/'summary.csv';p.write_text('score\n1\n')
            (root/'manifest.json').write_text(json.dumps({'schema_version':1,'files':{'summary.csv':{'sha256':digest(p)}}}))
            verify_bundle(root)
            p.write_text('score\n2\n')
            with self.assertRaises(ValueError):verify_bundle(root)

    def test_manifest_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'reports';root.mkdir()
            p=root.parent/'outside.csv';p.write_text('private')
            (root/'manifest.json').write_text(json.dumps({'schema_version':1,'files':{'../outside.csv':{'sha256':digest(p)}}}))
            with self.assertRaises(ValueError):verify_bundle(root)

if __name__=='__main__':unittest.main()
