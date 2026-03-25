"""Interactive or batch code summarization using a trained model."""

import argparse
import sys

from codeuctivity.inference import summarize
from codeuctivity.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize code snippets")
    parser.add_argument("--config", default="configs/full.yaml", help="Path to YAML config")
    parser.add_argument("--code", default=None, help="Code string to summarize (interactive if omitted)")
    parser.add_argument("--file", default=None, help="File containing code to summarize")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter_path = cfg["model"]["output_dir"]

    if args.file:
        code = open(args.file).read()
        print(summarize(code, adapter_path))
    elif args.code:
        print(summarize(args.code, adapter_path))
    else:
        print("Enter code (Ctrl+D to finish):")
        code = sys.stdin.read()
        print(summarize(code, adapter_path))


if __name__ == "__main__":
    main()
