"""Reuse the evaluator's existing downloads without changing pinned bytes."""
import argparse
import json
from _nyx_bundle.assets import stage

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', help='Directory containing the ten original HF task directories')
    parser.add_argument('--asset-map', help='JSON mapping a task to its download directory, or a manifest file path to an absolute downloaded file')
    parser.add_argument('--task', help='Prepare only this task; default: all ten')
    parser.add_argument('--copy', action='store_true', help='Copy identical files instead of symlinking')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(stage(args.task, args.asset_root, args.asset_map, args.copy, args.verify_only), indent=2))
if __name__ == '__main__':
    main()
