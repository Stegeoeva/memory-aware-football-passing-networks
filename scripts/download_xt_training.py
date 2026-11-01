"""Download the other 48 EURO 2024 matches, excluding our three evaluation games."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
COMMIT='4b73468fc5b0f1950f9f66fada70ad3a4f9327cb'
EXCLUDED={3942226,3942752,3943043}

def main():
    matches=json.loads((ROOT/'data/open-data/data/matches/55/282.json').read_text(encoding='utf-8'))
    ids=sorted(m['match_id'] for m in matches if m['match_id'] not in EXCLUDED)
    directory=ROOT/'data/xt_training/events'
    directory.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/'data/xt_training_manifest.json'
    previous=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    expected=previous.get('files',{}) if previous.get('commit')==COMMIT else {}
    def fetch(mid):
        path=directory/f'{mid}.json'
        correct=(path.exists() and str(mid) in expected and
                 hashlib.sha256(path.read_bytes()).hexdigest()==expected[str(mid)]['sha256'])
        if not correct:
            request=urllib.request.Request(f'https://raw.githubusercontent.com/hudl/open-data/{COMMIT}/data/events/{mid}.json',headers={'User-Agent':'FootballPassingNetworksResearch/1.0'})
            with urllib.request.urlopen(request,timeout=120) as response: data=response.read()
            assert isinstance(json.loads(data),list)
            if str(mid) in expected and hashlib.sha256(data).hexdigest()!=expected[str(mid)]['sha256']:
                raise ValueError(f'Pinned source checksum mismatch: {mid}')
            path.write_bytes(data)
        data=path.read_bytes()
        return str(mid),{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    with ThreadPoolExecutor(max_workers=4) as pool: files=dict(pool.map(fetch,ids))
    manifest={'repository':'https://github.com/hudl/open-data','commit':COMMIT,
              'competition_id':55,'season_id':282,'training_match_ids':ids,
              'excluded_evaluation_match_ids':sorted(EXCLUDED),'files':files}
    (ROOT/'data/xt_training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f"Training matches: {len(files)}; {sum(f['bytes'] for f in files.values())/1e6:.1f} MB. Evaluation games excluded.")

if __name__=='__main__': main()
