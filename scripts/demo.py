"""Run a local browser demo for the baseline and fine-tuned encoders."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from an_y.predict import predict_baseline, predict_zero_shot
from an_y.bert import predict_bert
from an_y.distilbert import predict_distilbert
from an_y.roberta import predict_roberta


PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>iSarcasmEval demo</title>
<style>
body{font:16px system-ui,sans-serif;max-width:760px;margin:48px auto;padding:0 20px;color:#15243a}
h1{font-size:28px}textarea,select,button{font:inherit;width:100%;box-sizing:border-box;margin:8px 0;padding:12px;border:1px solid #b9c5d3;border-radius:8px}
textarea{min-height:130px}button{background:#173d66;color:white;border:0;cursor:pointer}pre{white-space:pre-wrap;background:#f2f5f8;padding:16px;border-radius:8px}
.note{color:#52657a;font-size:14px}
</style>
<h1>iSarcasmEval sarcasm demo</h1>
<p>Enter an English sentence and choose a trained model.</p>
<label for="model">Model</label><select id="model">
<option value="baseline">TF-IDF + SVM</option>
<option value="zero-shot">Zero-shot, validation-selected prompt</option>
<option value="bert">BERT-base</option>
<option value="roberta">RoBERTa-base</option>
<option value="distilbert">DistilBERT</option>
</select>
<label for="text">Text</label><textarea id="text" maxlength="5000" placeholder="Type one English sentence…"></textarea>
<button id="run">Predict</button><pre id="result">Prediction appears here.</pre>
<p class="note">Labels: 0 = sincere, 1 = sarcastic. Model scores are not calibrated probabilities.</p>
<script>
document.querySelector('#run').onclick=async()=>{
 const result=document.querySelector('#result'); result.textContent='Running…';
 try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:document.querySelector('#model').value,text:document.querySelector('#text').value})});
 const data=await r.json(); result.textContent=r.ok?JSON.stringify(data,null,2):data.error;}
 catch(e){result.textContent=String(e);}
};
</script></html>"""


def make_handler(output_dir: Path, team_dir: Path):
    selected_architecture = "baseline"
    selected_path = team_dir / "selected_model.json"
    if selected_path.exists():
        selected_architecture = json.loads(
            selected_path.read_text(encoding="utf-8")
        ).get("architecture", "baseline")
    page = PAGE.replace(
        f'<option value="{selected_architecture}">',
        f'<option value="{selected_architecture}" selected>',
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16_000:
                    raise ValueError("Request must contain text under 5,000 characters.")
                payload = json.loads(self.rfile.read(length))
                text = str(payload.get("text", "")).strip()
                model = payload.get("model")
                if not text or len(text) > 5_000:
                    raise ValueError("Enter text with 1–5,000 characters.")
                if model == "baseline":
                    result = predict_baseline(text, output_dir)
                elif model == "zero-shot":
                    result = predict_zero_shot(text, "selected", output_dir)
                elif model in ("bert", "roberta", "distilbert"):
                    selection_path = team_dir / model / "selected_checkpoint.json"
                    if not selection_path.exists():
                        raise FileNotFoundError(
                            f"{model} selection is missing; train and package the model first."
                        )
                    checkpoint = Path(
                        json.loads(selection_path.read_text(encoding="utf-8"))["checkpoint"]
                    )
                    predictor = {
                        "bert": predict_bert,
                        "roberta": predict_roberta,
                        "distilbert": predict_distilbert,
                    }[model]
                    result = predictor(text, checkpoint)
                else:
                    raise ValueError("Unknown model selection.")
                self._json(200, result)
            except (ValueError, FileNotFoundError, json.JSONDecodeError, SystemExit) as error:
                self._json(400, {"error": str(error)})

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string, *args):
            print(f"demo: {format_string % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shared_review_20261017"))
    parser.add_argument("--team-dir", type=Path, default=Path("outputs/team"))
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), make_handler(args.output_dir, args.team_dir))
    print(f"Demo ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Demo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
