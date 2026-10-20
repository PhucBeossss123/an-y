"""Legacy combined entry point; use train_bert.py, train_roberta.py, or train_distilbert.py."""

import argparse
from pathlib import Path

from an_y.bert import BertConfig, run_bert
from an_y.distilbert import DistilBertConfig, run_distilbert
from an_y.roberta import RobertaConfig, run_roberta


MODELS = {
    "bert": (BertConfig, run_bert),
    "roberta": (RobertaConfig, run_roberta),
    "distilbert": (DistilBertConfig, run_distilbert),
}


def main(fixed_model: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_model is None:
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

    model = fixed_model or args.model
    config_type, run_model = MODELS[model]
    output_dir = args.output_dir or Path("outputs") / "team" / model
    config = config_type(
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seeds=tuple(args.seeds),
        mixed_precision=not args.no_mixed_precision,
    )
    metrics = run_model(
        data_dir=args.data_dir,
        split_file=args.split_file,
        split_manifest=args.split_manifest,
        output_dir=output_dir,
        config=config,
    )
    print(f"Saved {model} run to {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
