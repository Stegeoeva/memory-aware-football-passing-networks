"""Clear public notebooks while preserving executed local copies in an ignored folder."""
from pathlib import Path
import json
import shutil
import nbformat

ROOT=Path(__file__).resolve().parents[1]

def main():
    policy=json.loads((ROOT/'publication_allowlist.json').read_text(encoding='utf-8'))
    backup=ROOT/'.runtime/private_notebooks'
    changed=0
    for name in policy['source_notebooks']:
        path=ROOT/'notebooks'/name
        nb=nbformat.read(path,as_version=4)
        has_outputs=any(c.get('outputs') or c.get('attachments') or c.get('execution_count') is not None for c in nb.cells)
        if has_outputs:
            backup.mkdir(parents=True,exist_ok=True)
            target=backup/name
            if not target.exists():shutil.copy2(path,target)
            changed+=1
        nb.metadata.pop('widgets',None)
        for c in nb.cells:
            c.pop('attachments',None)
            if c.cell_type=='code':
                c.outputs=[];c.execution_count=None
                c.metadata.pop('execution',None)
        nbformat.write(nb,path)
    print(f'Source-only notebooks: {len(policy["source_notebooks"])}; cleared {changed} with private backups.')

if __name__=='__main__':main()
