"""Train a reproducible shared-split transformer run for the team comparison."""

import argparse
from pathlib import Path

from an_y.bert import BertConfig, run_bert


MODELS = {
    "bert": (
        "bert-base-uncased",
        "86b5e0934494bd15c9632b12f734a8a67f723594",
    ),
    "roberta": (
        "FacebookAI/roberta-base",
        "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
    ),
    "distilbert": (
        "distilbert/distilbert-base-uncased",
        "12040accade4e8a0f71eabdb258fecc2e7e948be",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--split-file", type=Path,
        default=Path("data/splits/shared_split_seed42.csv"),
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("data/splits/manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--no-mixed-precision", action="store_true")
    args = parser.parse_args()

    model_name, revision = MODELS[args.model]
    output_dir = args.output_dir or Path("outputs") / "team" / args.model
    config = BertConfig(
        model_name=model_name,
        revision=revision,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seeds=tuple(args.seeds),
        mixed_precision=not args.no_mixed_precision,
    )
    metrics = run_bert(
        data_dir=args.data_dir,
        split_file=args.split_file,
        split_manifest=args.split_manifest,
        output_dir=output_dir,
        config=config,
    )
    print(f"Saved {args.model} run to {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
