"""Local web demo for the project's DBpedia text classifiers."""
from __future__ import annotations

import csv
import json
import mimetypes
import sys
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]
MODELS = {
    "svm": {"name": "Linear SVM", "path": ROOT / "SVM/results_dbpedia_svm/linear_svc_tfidf.joblib", "metrics": ROOT / "SVM/results_dbpedia_svm/metrics.json"},
    "bert": {"name": "BERT Base", "path": ROOT / "BERT/results_dbpedia/checkpoint-35000", "metrics": ROOT / "BERT/results_dbpedia/metrics.json"},
    "gpt2": {"name": "GPT-2", "path": ROOT / "GPT/results_dbpedia_gpt2", "metrics": ROOT / "GPT/results_dbpedia_gpt2/metrics.json"},
}
_loaded: dict[str, object] = {}


def get_model(key: str):
    if key in _loaded:
        return _loaded[key]
    item = MODELS[key]
    if not item["path"].exists():
        raise FileNotFoundError(f"Chưa tìm thấy tệp mô hình {item['name']}: {item['path']}")
    if key == "svm":
        from joblib import load
        # The training script originally ran as __main__, so the saved
        # vectorizer can resolve its Counter global through the training module.
        from SVM import tfidf_svm
        tfidf_svm.Counter = Counter
        # The artifact was serialized while the training file was __main__.
        # Restore the globals used by that vectorizer for joblib compatibility.
        training_globals = sys.modules["__main__"].__dict__
        for name in ("Counter", "TOKEN_PATTERN", "np", "math", "csr_matrix", "vstack"):
            training_globals[name] = getattr(tfidf_svm, name)
        bundle = load(item["path"])
        model = (bundle["vectorizer"], bundle["classifier"], bundle["class_names"])
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_path = item["path"]
        if key == "gpt2" and not any(
            (model_path / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin")
        ):
            from transformers.trainer_utils import get_last_checkpoint

            checkpoint = get_last_checkpoint(str(model_path))
            if checkpoint is None:
                raise FileNotFoundError(
                    f"Chưa tìm thấy model GPT-2 hoặc checkpoint trong {model_path}"
                )
            model_path = Path(checkpoint)
        tokenizer_source = (
            "bert-base-uncased" if key == "bert" else "openai-community/gpt2"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, local_files_only=(key == "bert")
        )
        model_obj = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model_obj.eval()
        model = (tokenizer, model_obj, torch)
    _loaded[key] = model
    return model


def classify(key: str, text: str) -> dict:
    if not text.strip():
        raise ValueError("Vui lòng nhập nội dung văn bản.")
    model = get_model(key)
    if key == "svm":
        vectorizer, classifier, labels = model
        prediction = int(classifier.predict(vectorizer.transform([text]))[0])
        return {"label": labels[prediction], "confidence": None}
    tokenizer, classifier, torch = model
    if key == "gpt2" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        classifier.config.pad_token_id = tokenizer.pad_token_id
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        probabilities = torch.softmax(classifier(**tokens).logits, dim=-1)[0]
    index = int(probabilities.argmax())
    label = CLASS_NAMES[index] if key == "bert" else classifier.config.id2label.get(index, CLASS_NAMES[index])
    return {"label": label, "confidence": float(probabilities[index])}


def model_catalog() -> list[dict]:
    result = []
    for key, item in MODELS.items():
        metrics = None
        if item["metrics"].exists():
            try:
                metrics = json.loads(item["metrics"].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        test = metrics.get("test", {}) if metrics else {}
        matrix_path = item["metrics"].parent / "confusion_matrix_test.csv"
        matrix = None
        if matrix_path.exists():
            with matrix_path.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
            matrix = {"labels": rows[0][1:], "values": [[int(v) for v in row[1:]] for row in rows[1:]]}
        report_path = item["metrics"].parent / "classification_report_test.txt"
        report = report_path.read_text(encoding="utf-8") if report_path.exists() else None
        result.append({"id": key, "name": item["name"], "accuracy": test.get("accuracy"), "precision": test.get("precision_macro", test.get("precision")), "recall": test.get("recall_macro", test.get("recall")), "f1": test.get("f1_macro", test.get("f1")), "samples": metrics.get("sample_counts", {}).get("test") if metrics else None, "matrix": matrix, "report": report, "available": item["path"].exists()})
    return result


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/models":
            return self.send_json({"models": model_catalog()})
        target = (ROOT / unquote(path.lstrip("/"))).resolve() if path != "/" else ROOT / "index.html"
        if ROOT not in target.parents or not target.is_file():
            self.send_error(404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if urlparse(self.path).path != "/api/classify":
            return self.send_error(404)
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            result = classify(payload.get("model", "svm"), payload.get("text", ""))
            self.send_json(result)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": f"Không thể chạy mô hình: {exc}"}, 500)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    address = ("127.0.0.1", 8000)
    print(f"Mở trang tại http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, Handler).serve_forever()
