# Team transformer comparison

All three encoders use the official shared train/validation IDs and official test, with seeds 42, 123, and 2026. Epochs are selected per seed on validation Macro-F1, with sarcastic-class F1 as tie-break. Test scores were not used to select the demo model.

| Model | Validation Macro-F1 (mean ± sample SD) | Validation sarcastic F1 | Test Macro-F1 | Test sarcastic F1 |
|---|---:|---:|---:|---:|
| bert-base-uncased | 0.6002 ± 0.0125 | 0.3798 ± 0.0513 | 0.5870 ± 0.0124 | 0.3234 ± 0.0200 |
| FacebookAI/roberta-base | 0.6151 ± 0.0386 | 0.3777 ± 0.0832 | 0.6168 ± 0.0295 | 0.3542 ± 0.0723 |
| distilbert/distilbert-base-uncased | 0.5597 ± 0.0438 | 0.2752 ± 0.0962 | 0.5717 ± 0.0236 | 0.2631 ± 0.0630 |

Demo model selected on validation only: **FacebookAI/roberta-base**, seed `42`.
Per-seed ID-only predictions and metrics are stored in each model folder. Checkpoints remain under the ignored `outputs/team/` directory.

Run `python scripts/demo.py` after training to launch the local demo.
