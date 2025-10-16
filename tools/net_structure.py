#!/usr/bin/env python3
"""Utility: load a checkpoint (.pth) and print the model state_dict keys.

Usage:
    python tools/net_structure.py --ckpt checkpoints/dust3r_SL_224_pattern_decoder6/checkpoint-final.pth
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(__doc__)
    # Provide a sensible default so the script can be run without args.
    default_ckpt = '/data3/hanning/dust3r1/checkpoints/dust3r_SL_224/checkpoint-best.pth'
    parser.add_argument('--ckpt', default=default_ckpt, help=f'path to checkpoint .pth file (default: {default_ckpt})')
    args = parser.parse_args()

    ckpt_path = args.ckpt
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model' not in ckpt:
        print('Warning: checkpoint has no "model" key. Available keys:', list(ckpt.keys()))
        return
    keys = sorted(list(ckpt['model'].keys()))
    print('Number of model keys:', len(keys))
    # print first N keys for overview
    N = 200
    print(f'First {min(N, len(keys))} keys:')
    for k in keys[:N]:
        print(k)

    # report pattern related keys
    pattern_keys = [k for k in keys if k.startswith('pattern') or k.startswith('pattern_embed')]
    print('\nPattern-related key count:', len(pattern_keys))
    if pattern_keys:
        print('Pattern keys:')
        for k in pattern_keys:
            print(k)


if __name__ == '__main__':
    main()
