"""Optional fixed-asset downloader; existing downloads can be reused instead."""
import argparse
import json
from _nyx_bundle.assets import ROOT, records, stage
REPO = 'Lastforever16/robosyn-ten-task-models'
REVISION = 'da23c29b897034d0ca44f703d09985f21d629def'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default=REPO)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo_id, repo_type='model', revision=args.revision,
                          local_dir=ROOT, allow_patterns=[r['path'] for r in records()])
    print(json.dumps(stage(verify_only=True), indent=2))
if __name__ == '__main__':
    main()
