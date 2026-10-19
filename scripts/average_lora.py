# -*- coding: utf-8 -*-
"""Uniform average of train-jev LoRA checkpoints (adapter weights + decision head).

StartLux averages 3 seeds at 0.8B and 4 seeds at 2B–27B after Stage 2. This script only averages
checkpoints that share an adapter config, verbalizer ids and attention mode. It does not train.

    python scripts/average_lora.py --out runs/jev-4b-instant-avg runs/seed0/best runs/seed1/best runs/seed2/best
"""
import argparse
import json

from trainjev.average import average_checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args(argv)
    print(json.dumps(average_checkpoints(args.checkpoints, args.out)))


if __name__ == "__main__":
    main()
