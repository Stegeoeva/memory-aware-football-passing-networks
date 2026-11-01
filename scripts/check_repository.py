"""Check the public allowlist, stripped notebooks and aggregate-only publication."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from urllib.parse import unquote,urlsplit
import nbformat
from PIL import Image
from publication_policy import ROOT,public_paths

def publication_files(root=ROOT):
    yield from public_paths(root)

def check_links(path,text,included):
    refs=re.findall(r'(?:href|src)=["\']([^"\']+)["\']',text)
    if path.suffix in {'.md','.ipynb'}:refs+=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
    count=0
    for ref in refs:
        if '{' in ref or ref.startswith(('#','data:','mailto:')):continue
        parsed=urlsplit(ref.strip('<>'))
        if parsed.scheme or parsed.netloc:continue
        target=(path.parent/unquote(parsed.path)).resolve()
        assert target in included,('link is not in public package',path.relative_to(ROOT),ref)
        count+=1
    return count

def main():
    files=list(public_paths());included={p.resolve() for p in files}
    policy=json.loads((ROOT/'publication_allowlist.json').read_text(encoding='utf-8'))
    relative_included={p.relative_to(ROOT).as_posix() for p in files}
    # Only inspect this checkout's Git index, never an unrelated parent repository.
    try:
        top=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=ROOT,capture_output=True,text=True,check=False)
        if top.returncode==0 and Path(top.stdout.strip()).resolve()==ROOT.resolve():
            tracked=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,check=True)
            names={n for n in tracked.stdout.decode('utf-8').split('\0') if n}
            assert names<=relative_included,('tracked files outside public allowlist',sorted(names-relative_included))
    except FileNotFoundError:
        pass  # The ZIP can be checked without installing Git.
    links=notebooks=cells=figures=0
    for path in files:
        rel=path.relative_to(ROOT).as_posix()
        assert path.stat().st_size<25*1024**2,('unexpected large public file',rel)
        assert path.suffix not in {'.csv','.gz','.npz','.npy','.pkl','.pickle','.joblib','.graphml','.parquet'},rel
        if rel.startswith('results/'):
            assert rel.startswith('results/public/'),rel
            assert path.suffix in {'.png','.svg','.html','.json'},rel
        if rel.startswith('data/'):
            assert rel in {'data/manifest.json','data/xt_training_manifest.json'},rel
        if path.suffix in {'.py','.md','.html','.json','.ipynb','.yml','.txt'}:
            text=path.read_text(encoding='utf-8')
            assert not re.search(r'[Α-Ωα-ω]{3}',text),('non-English prose',rel)
            assert not re.search(r'C:[\\/]+Users[\\/]|file:/{3}',text,re.I),('local user path',rel)
            assert not re.search(r'ghp_[A-Za-z0-9]{36}|-----BEGIN [A-Z ]*PRIVATE KEY-----',text),('possible secret',rel)
            if path.suffix=='.py':ast.parse(text,filename=rel)
            if path.suffix in {'.md','.html'}:links+=check_links(path,text,included)
            if path.suffix=='.html':assert '<html lang="en">' in text,rel
        if path.suffix=='.ipynb':
            nb=nbformat.read(path,as_version=4);nbformat.validate(nb);notebooks+=1
            assert not nb.metadata.get('widgets'),('embedded widget state',rel)
            for c in nb.cells:
                assert not c.get('attachments'),('notebook attachment',rel)
                if c.cell_type=='markdown':links+=check_links(path,c.source,included)
                if c.cell_type=='code':
                    cells+=1
                    assert c.execution_count is None and not c.outputs,('notebook outputs must be stripped',rel)
        if path.suffix=='.png' and path.parent.name=='figures':
            figures+=1
            with Image.open(path) as im:im.verify()
            ET.parse(path.with_suffix('.svg'))
    assert notebooks==len(policy['source_notebooks'])==4
    assert figures==len(policy['aggregate_figures'])==12
    record=json.loads((ROOT/'provenance/public_release.json').read_text(encoding='utf-8'))
    for name,expected in record['aggregate_figure_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    result={'status':'passed','publication_profile':'public-code-and-aggregate-analysis',
        'public_files':len(files),'public_links_checked':links,'source_notebooks':notebooks,
        'code_cells_with_outputs':0,'source_code_cells':cells,'aggregate_figure_pairs':figures,
        'redistributed_event_tables':0,'trained_parameter_files':0,'largest_file_bytes':max(p.stat().st_size for p in files),
            'legal_scope':'Packaging checks, not a legal opinion or written permission from the data owner.'}
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':main()
