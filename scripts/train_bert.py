"""Run the fixed BERT-base three-seed experiment."""

import argparse
import json
from pathlib import Path

from an_y.bert import BertConfig, run_bert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/shared_split_seed42.csv"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert"))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument(
        "--revision",
        default="86b5e0934494bd15c9632b12f734a8a67f723594",
        help="Hugging Face model revision",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()

    config = BertConfig(
        model_name=args.model_name,
        revision=args.revision,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seeds=tuple(args.seeds),
        mixed_precision=not args.no_mixed_precision,
    )
    result = run_bert(
        data_dir=args.data_dir,
        split_file=args.split_file,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        config=config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
