"""Train and predict with BERT-base on iSarcasmEval."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from .data import load_shared_split, load_test
from .metrics import summarize


@dataclass(frozen=True)
class BertConfig:
    model_name: str = "bert-base-uncased"
    revision: str | None = "86b5e0934494bd15c9632b12f734a8a67f723594"
    max_length: int = 128
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    learning_rate: float = 2e-5
    decision_threshold: float = 0.535
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seeds: tuple[int, ...] = (42, 123, 2026)
    mixed_precision: bool = True


class EncodedTexts(Dataset):
    """Tokenized text and labels only; annotations never enter the tokenizer."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = frame["label"].astype(int).tolist()
        self.sample_ids = frame["sample_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _collate(tokenizer):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([item.pop("labels") for item in batch], dtype=torch.long)
        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded

    return collate


def _evaluate(model, loader, device, decision_threshold: float = 0.535) -> tuple[dict, np.ndarray]:
    model.eval()
    labels = []
    predictions = []
    for batch in loader:
        gold = batch.pop("labels").numpy()
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            scores = torch.softmax(model(**batch).logits, dim=-1)[:, 1]
        predictions.extend((scores >= decision_threshold).long().cpu().numpy().tolist())
        labels.extend(gold.tolist())
    return summarize(labels, predictions), np.asarray(predictions, dtype=np.int64)


def _write_predictions(frame: pd.DataFrame, predictions: np.ndarray, path: Path) -> None:
    pd.DataFrame({
        "sample_id": frame["sample_id"].astype(str),
        "y_true": frame["label"].astype(int),
        "y_pred": predictions.astype(int),
    }).to_csv(path, index=False)


def _score_pair(metrics: dict) -> tuple[float, float]:
    return metrics["macro_f1"], metrics["f1_sarcastic"]


def _train_seed(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    tokenizer,
    config: BertConfig,
    seed: int,
    output_dir: Path,
) -> dict:
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate = _collate(tokenizer)
    fit_dataset = EncodedTexts(fit, tokenizer, config.max_length)
    validation_dataset = EncodedTexts(validation, tokenizer, config.max_length)
    generator = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model_args = {"revision": config.revision} if config.revision else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "sincere", 1: "sarcastic"},
        label2id={"sincere": 0, "sarcastic": 1},
        **model_args,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    updates_per_epoch = int(np.ceil(len(fit_loader) / config.gradient_accumulation_steps))
    total_steps = updates_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_dir = output_dir / f"seed_{seed}"
    checkpoint_dir = run_dir / "best_checkpoint"
    best_score = (-1.0, -1.0)
    best_epoch = 0
    best_metrics = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(fit_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            group_start = ((step - 1) // config.gradient_accumulation_steps) * config.gradient_accumulation_steps
            group_size = min(
                config.gradient_accumulation_steps, len(fit_loader) - group_start
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss / group_size
            scaler.scale(loss).backward()
            if step % config.gradient_accumulation_steps == 0 or step == len(fit_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation_metrics, validation_predictions = _evaluate(
            model, validation_loader, device, config.decision_threshold
        )
        score = _score_pair(validation_metrics)
        print(
            f"seed={seed} epoch={epoch} validation_macro_f1={score[0]:.4f} "
            f"validation_f1_sarcastic={score[1]:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = validation_metrics
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_config = asdict(config)
            checkpoint_config["seeds"] = list(config.seeds)
            (checkpoint_dir / "training_config.json").write_text(
                json.dumps(checkpoint_config, indent=2) + "\n", encoding="utf-8"
            )
            _write_predictions(
                validation, validation_predictions, run_dir / "validation_predictions.csv"
            )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "selected_epoch": best_epoch,
        "validation": best_metrics,
        "checkpoint": str(checkpoint_dir),
    }


def _predict_checkpoint(
    frame: pd.DataFrame,
    checkpoint: Path,
    config: BertConfig,
) -> tuple[dict, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    dataset = EncodedTexts(frame, tokenizer, config.max_length)
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=_collate(tokenizer),
    )
    config_path = checkpoint / "training_config.json"
    training_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    metrics, predictions = _evaluate(
        model, loader, device,
        float(training_config.get("decision_threshold", config.decision_threshold)),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, predictions


def _mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": float(np.mean(values)),
        "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def run_bert(
    data_dir: str | Path = "data/raw",
    split_file: str | Path = "data/splits/shared_split_seed42.csv",
    split_manifest: str | Path = "data/splits/manifest.json",
    output_dir: str | Path = "outputs/team/bert",
    config: BertConfig | None = None,
) -> dict:
    """Fine-tune this model for each seed; select checkpoints on validation."""
    config = config or BertConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fit, validation = load_shared_split(data_dir, split_file, split_manifest)
    test = load_test(data_dir)
    split_metadata = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    tokenizer_args = {"revision": config.revision} if config.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, **tokenizer_args)
    resolved_revision = tokenizer.init_kwargs.get("_commit_hash") or config.revision
    results = []
    for seed in config.seeds:
        result = _train_seed(fit, validation, tokenizer, config, seed, output)
        test_metrics, test_predictions = _predict_checkpoint(
            test, Path(result["checkpoint"]), config
        )
        result["official_test"] = test_metrics
        _write_predictions(
            test,
            test_predictions,
            output / f"seed_{seed}" / "official_test_predictions.csv",
        )
        results.append(result)

    metrics = {
        "model_name": config.model_name,
        "revision_requested": config.revision or "main at run time",
        "revision_resolved": resolved_revision,
        "source_revision": split_metadata["source_revision"],
        "source_file_checksums": {
            name: value["sha256"]
            for name, value in split_metadata["source_files"].items()
        },
        "split_sha256": split_metadata["split_sha256"],
        "config": asdict(config),
        "train_rows": int(len(fit)),
        "validation_rows": int(len(validation)),
        "official_test_rows": int(len(test)),
        "selection": "best validation Macro-F1; ties resolved by sarcastic-class F1",
        "seeds": results,
        "validation_summary": {
            metric: _mean_std([run["validation"][metric] for run in results])
            for metric in ("macro_f1", "f1_sarcastic")
        },
        "official_test_summary": {
            metric: _mean_std([run["official_test"][metric] for run in results])
            for metric in ("macro_f1", "f1_sarcastic")
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


@lru_cache(maxsize=1)
def _load_prediction_model(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    config_path = checkpoint / "training_config.json"
    training_config = {}
    if config_path.exists():
        training_config = json.loads(config_path.read_text(encoding="utf-8"))
    return (
        tokenizer,
        model,
        device,
        int(training_config.get("max_length", 128)),
        training_config.get("model_name", "bert-base-uncased"),
        float(training_config.get("decision_threshold", 0.535)),
    )


def predict_bert(text: str, checkpoint_dir: str | Path) -> dict:
    """Predict one text with a fine-tuned encoder, without a probability claim."""
    tokenizer, model, device, max_length, method, decision_threshold = _load_prediction_model(
        str(Path(checkpoint_dir).resolve())
    )
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.no_grad():
        score = torch.softmax(model(**encoded).logits, dim=-1)[0, 1].item()
        prediction = int(score >= decision_threshold)
    return {
        "method": method,
        "y_pred": prediction,
        "label": "sarcastic" if prediction == 1 else "sincere",
    }
