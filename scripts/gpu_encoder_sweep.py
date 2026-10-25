"""Run a validation-only learning-rate sweep, then test one winner per encoder."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from transformers import AutoTokenizer

from an_y.data import load_shared_split, load_test
from an_y.bert import BertConfig, _predict_checkpoint as predict_bert, _train_seed as train_bert
from an_y.roberta import RobertaConfig, _predict_checkpoint as predict_roberta, _train_seed as train_roberta
from an_y.distilbert import DistilBertConfig, _predict_checkpoint as predict_distilbert, _train_seed as train_distilbert


MODELS = {
    "bert": (BertConfig, train_bert, predict_bert),
    "roberta": (RobertaConfig, train_roberta, predict_roberta),
    "distilbert": (DistilBertConfig, train_distilbert, predict_distilbert),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/shared_split_seed42.csv"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/gpu_encoder_sweep"))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-5, 2e-5, 3e-5])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently run the sweep on CPU")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    fit, validation = load_shared_split(args.data_dir, args.split_file, args.split_manifest)
    test = load_test(args.data_dir)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    results = {}

    for model_key, (config_type, train_fn, predict_fn) in MODELS.items():
        base = config_type()
        tokenizer_args = {"revision": base.revision} if base.revision else {}
        tokenizer = AutoTokenizer.from_pretrained(base.model_name, **tokenizer_args)
        candidates = []
        for learning_rate in args.learning_rates:
            config = replace(
                base,
                learning_rate=learning_rate,
                epochs=args.epochs,
                seeds=(args.seed,),
                train_batch_size=args.train_batch_size,
                mixed_precision=True,
            )
            candidate_dir = args.output_dir / model_key / f"lr_{learning_rate:g}"
            print(
                f"\n[{model_key}] lr={learning_rate:g}, epochs={args.epochs}, "
                f"seed={args.seed}, batch={args.train_batch_size}, AMP=on",
                flush=True,
            )
            result = train_fn(fit, validation, tokenizer, config, args.seed, candidate_dir)
            result.update({"learning_rate": learning_rate, "config": asdict(config)})
            candidates.append(result)

        winner = max(
            candidates,
            key=lambda item: (
                item["validation"]["macro_f1"],
                item["validation"]["f1_sarcastic"],
                -abs(item["learning_rate"] - base.learning_rate),
            ),
        )
        print(
            f"[{model_key}] validation winner lr={winner['learning_rate']:g}: "
            f"Macro-F1={winner['validation']['macro_f1']:.4f}, "
            f"sarcastic-F1={winner['validation']['f1_sarcastic']:.4f}",
            flush=True,
        )
        winner_config = replace(base, **{
            key: value for key, value in winner["config"].items()
            if key in {field for field in base.__dataclass_fields__}
        })
        test_metrics, _ = predict_fn(test, Path(winner["checkpoint"]), winner_config)
        winner["official_test"] = test_metrics
        results[model_key] = {"candidates": candidates, "winner": winner}
        torch.cuda.empty_cache()

    report = {
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "selection": "learning rate selected by validation Macro-F1; ties by sarcastic F1, then nearest default; official test evaluated only for the selected learning rate per model",
        "train_rows": len(fit),
        "validation_rows": len(validation),
        "official_test_rows": len(test),
        "source_revision": manifest["source_revision"],
        "source_file_checksums": {
            name: record["sha256"]
            for name, record in manifest["source_files"].items()
        },
        "split_sha256": manifest["split_sha256"],
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
