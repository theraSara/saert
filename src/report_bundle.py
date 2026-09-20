"""Portable, checksum-verified aggregate reports. No corpus or model dependencies."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_bundle(root=None):
    root = Path(root or ROOT/'reports').resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unexpected report bundle version')
    for name, record in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Unsafe or missing report path: {name}')
        if digest(path) != record['sha256']:
            raise ValueError(f'Report checksum mismatch: {name}')
    return manifest

def read_summary(section, name, root=None):
    import pandas as pd
    root = Path(root or ROOT/'reports')
    manifest = verify_bundle(root)
    relative = f'{section}/{name}.csv'
    if relative not in manifest['files']:
        raise ValueError(f'Unlisted summary: {relative}')
    return pd.read_csv(root/relative)
