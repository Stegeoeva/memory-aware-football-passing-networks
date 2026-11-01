"""Package only the public allowlist; local data and trained outputs stay private."""
from pathlib import Path
import argparse
import json
import zipfile
from check_repository import ROOT, main, publication_files

def package(output):
    checks=main()
    (ROOT/'results/public/publication_check.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    output=Path(output).resolve()
    files=sorted(publication_files())
    assert output.suffix=='.zip'
    assert not output.is_relative_to(ROOT),'Write the archive outside the project directory.'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:archive.write(path,Path(ROOT.name)/path.relative_to(ROOT))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(files)
    print(json.dumps({'archive':str(output),'files':len(files),'bytes':output.stat().st_size,
        'content_check':checks['status']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'Football_Passing_Networks_PUBLIC.zip')
    package(parser.parse_args().output)
