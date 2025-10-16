#!/usr/bin/env python3
"""Count files in each subfolder of a root directory and report folders with fewer than a threshold files.

Default root: /data3/hanning/datasets/scannetpp_SLpattern
Default threshold: 20

Usage:
    python tools/count_scannet_folders.py
    python tools/count_scannet_folders.py --root /path/to/root --threshold 10 --ext .jpg
"""
import os
import argparse


def count_files_in_subfolders(root_dir, threshold=20, exts=None):
    if exts is None:
        exts = None
    results = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    subfolders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    for sub in subfolders:
        subpath = os.path.join(root_dir, sub)
        files = [f for f in os.listdir(subpath) if os.path.isfile(os.path.join(subpath, f))]
        if exts:
            files = [f for f in files if os.path.splitext(f)[1].lower() in exts]
        count = len(files)
        results.append((sub, count))
    # filter
    below = [(s,c) for s,c in results if c < threshold]
    return results, below


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', default='/data3/hanning/datasets/scannetpp_SLpattern', help='root directory containing scene subfolders')
    parser.add_argument('--threshold', type=int, default=20, help='threshold for "too few" files')
    parser.add_argument('--ext', type=str, default=None, help='only count files with this extension (e.g. .jpg). Default: count all files')
    args = parser.parse_args()
    exts = None
    if args.ext:
        exts = [args.ext.lower()]
    results, below = count_files_in_subfolders(args.root, threshold=args.threshold, exts=exts)
    total = len(results)
    print(f'Found {total} subfolders under {args.root}')
    print(f'{len(below)} folders have fewer than {args.threshold} files:')
    for s,c in below:
        print(f'  {s}: {c}')
    # summary optionally print top few
    if below:
        print('\nSample of folders below threshold:')
        for s,c in below[:20]:
            print(s, c)

if __name__ == '__main__':
    main()
