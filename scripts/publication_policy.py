"""Explicit public-file policy: source code and curated aggregate analysis only."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    'README.md','LICENSE','THIRD_PARTY_NOTICES.md','.gitignore','.gitattributes',
    'requirements.txt','requirements-lock.txt','requirements-memory.txt',
    'requirements-memory-lock.txt','publication_allowlist.json',
}

def public_paths(root=ROOT):
    root=Path(root)
    policy=json.loads((root/'publication_allowlist.json').read_text(encoding='utf-8'))
    selected={root/p for p in ROOT_FILES}
    for folder in ['src','scripts','tests']:
        selected.update(p for p in (root/folder).rglob('*.py') if '__pycache__' not in p.parts)
    selected.update((root/'docs').glob('*.md'))
    selected.update(root/p for p in policy['fixed_files'])
    selected.update(root/'notebooks'/name for name in policy['source_notebooks'])
    for item in policy['aggregate_figures']:
        for suffix in ['.png','.svg']:
            selected.add((root/item['public_stem']).with_suffix(suffix))
    for path in sorted(selected):
        assert path.is_relative_to(root),path
        assert path.is_file(),f'Missing public file: {path.relative_to(root)}'
        yield path

def relative_public_paths(root=ROOT):
    return {p.relative_to(root).as_posix() for p in public_paths(root)}
