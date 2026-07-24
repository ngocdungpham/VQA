
import argparse
import json
import math
import os
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("LLAVA_DEFER_VISION_TOWER", "1")

import torch
from PIL import Image
from peft import PeftModel
import transformers
from transformers.utils.quantization_config import QuantizationMethod

from llava import conversation as conversation_lib
from llava import LlavaLlamaForCausalLM
from llava.model.llava import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)


TAG_RE = {
    "answer": re.compile(r"<answer>\s*(.*?)(?=\s*<reason>|\s*<location>|$)", re.I | re.S),
    "reason": re.compile(r"<reason>\s*(.*?)(?=\s*<location>|$)", re.I | re.S),
    "location": re.compile(r"<location>\s*(.*?)(?=\s*###|$)", re.I | re.S),
}
BOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def extract_tag(text, name):
    match = TAG_RE[name].search(text or "")
    return match.group(1).strip() if match else ""


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    return " ".join(text.split())


def token_f1(prediction, target):
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(target_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_text(text):
    return extract_tag(text, "answer")


def choice_labels(text):
    """Extract option labels such as A/B/C from the structured answer."""
    answer = answer_text(text).upper()
    labels = set(re.findall(r"\b([A-H])\s*[\.\)]", answer))
    if not labels:
        labels = set(re.findall(r"(?:^|[,;/])\s*([A-H])(?=\s*(?:[,;/]|$))", answer))
    return labels


def yes_no_label(text):
    answer = normalize_text(answer_text(text))
    if re.search(r"\byes\b", answer):
        return "yes"
    if re.search(r"\bno\b", answer):
        return "no"
    return None


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def rouge_l_f1(prediction, target):
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    previous = [0] * (len(target_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for index, target_token in enumerate(target_tokens, start=1):
            if pred_token == target_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(target_tokens)
    return safe_div(2 * precision * recall, precision + recall)


def ngram_counts(tokens, order):
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def cider_scores(predictions, targets, max_order=4):
    """Compute corpus-IDF CIDEr with one reference per sample."""
    if not predictions:
        return []
    reference_tokens = [normalize_text(text).split() for text in targets]
    prediction_tokens = [normalize_text(text).split() for text in predictions]
    document_frequency = [Counter() for _ in range(max_order)]
    for tokens in reference_tokens:
        for order in range(1, max_order + 1):
            for ngram in ngram_counts(tokens, order):
                document_frequency[order - 1][ngram] += 1
    log_document_count = math.log(max(1, len(targets)))

    def vector(counts, order):
        values = {}
        for ngram, count in counts.items():
            df = document_frequency[order - 1].get(ngram, 0)
            idf = log_document_count - math.log(max(1, df))
            values[ngram] = count * idf
        norm = math.sqrt(sum(value * value for value in values.values()))
        return values, norm

    scores = []
    for pred_tokens, ref_tokens in zip(prediction_tokens, reference_tokens):
        similarities = []
        for order in range(1, max_order + 1):
            pred_vector, pred_norm = vector(ngram_counts(pred_tokens, order), order)
            ref_vector, ref_norm = vector(ngram_counts(ref_tokens, order), order)
            if pred_norm == 0 or ref_norm == 0:
                similarities.append(0.0)
                continue
            dot = sum(value * ref_vector.get(ngram, 0.0) for ngram, value in pred_vector.items())
            similarities.append(dot / (pred_norm * ref_norm))
        scores.append(10.0 * sum(similarities) / max_order)
    return scores


def parse_boxes(text):
    location = extract_tag(text, "location")
    return [[float(value) for value in match] for match in BOX_RE.findall(location)]


def box_iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def location_score(predicted, target):
    pred_boxes = parse_boxes(predicted)
    target_boxes = parse_boxes(target)
    if not target_boxes:
        return None
    if not pred_boxes:
        return 0.0
    # Each ground-truth box receives its best predicted match.
    return sum(max(box_iou(gt, pred) for pred in pred_boxes) for gt in target_boxes) / len(target_boxes)


def score_prediction(prediction, target):
    pred_answer = extract_tag(prediction, "answer")
    target_answer = extract_tag(target, "answer")
    iou = location_score(prediction, target)
    return {
        "answer_exact": float(normalize_text(pred_answer) == normalize_text(target_answer)),
        "answer_token_f1": token_f1(pred_answer, target_answer),
        "location_iou": iou,
        "location_iou_50": None if iou is None else float(iou >= 0.5),
    }


def format_prompt(item, image_token_len):
    human_messages = [x["value"] for x in item["conversations"] if x["from"] == "human"]
    if not human_messages:
        raise ValueError(f"Record {item.get('row_id')} has no human message")
    image_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    image_tokens = DEFAULT_IM_START_TOKEN + image_tokens + DEFAULT_IM_END_TOKEN
    human = human_messages[0].replace(DEFAULT_IMAGE_TOKEN, image_tokens)
    header = f"{conversation_lib.default_conversation.system} {item['type_prompt']}\n\n"
    return f"{header}### Human: {human}\n### Assistant: "


def load_model(base_model, adapter, device):
    compute_dtype = torch.float16
    import transformers.modeling_utils as modeling_utils

    original_dispatch_model = modeling_utils.dispatch_model

    def dispatch_model_with_quant_hooks(model, *args, **kwargs):
        kwargs["force_hooks"] = True
        return original_dispatch_model(model, *args, **kwargs)

    modeling_utils.dispatch_model = dispatch_model_with_quant_hooks
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=["mm_projector"],
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        quantization_config=quantization_config,
    )
    model.is_loaded_in_4bit = True
    model.is_quantized = True
    model.quantization_method = QuantizationMethod.BITS_AND_BYTES

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        adapter, use_fast=False, padding_side="right"
    )
    vision = model.model.initialize_vision_modules(
        vision_tower="openai/clip-vit-large-patch14",
        mm_vision_select_layer=-2,
        pretrain_mm_mlp_adapter=None,
    )
    model.model.vision_tower[0].to(dtype=torch.float16, device=device)
    model.config.mm_use_im_start_end = True
    model.initialize_vision_tokenizer(
        mm_use_im_start_end=True,
        tokenizer=tokenizer,
        device=device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    )
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, vision["image_processor"], vision["image_token_len"]


def closed_ended_metrics(items):
    targets = [yes_no_label(row["target"]) for row in items]
    predictions = [yes_no_label(row["prediction"]) for row in items]
    tp = sum(pred == "yes" and target == "yes" for pred, target in zip(predictions, targets))
    fp = sum(pred == "yes" and target == "no" for pred, target in zip(predictions, targets))
    fn = sum(pred != "yes" and target == "yes" for pred, target in zip(predictions, targets))
    tn = sum(pred == "no" and target == "no" for pred, target in zip(predictions, targets))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "records": len(items),
        "positive_label": "yes",
        "positive_support": sum(target == "yes" for target in targets),
        "negative_support": sum(target == "no" for target in targets),
        "prediction_coverage": safe_div(sum(pred is not None for pred in predictions), len(items)),
        "precision": precision,
        "recall_sensitivity": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "specificity": safe_div(tn, tn + fp),
        "accuracy": safe_div(tp + tn, len(items)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def single_choice_metrics(items):
    targets = [choice_labels(row["target"]) for row in items]
    predictions = [choice_labels(row["prediction"]) for row in items]
    correct = sum(
        len(target) == 1 and len(prediction) == 1 and target == prediction
        for prediction, target in zip(predictions, targets)
    )
    return {
        "records": len(items),
        "accuracy": safe_div(correct, len(items)),
        "prediction_coverage": safe_div(
            sum(len(prediction) == 1 for prediction in predictions), len(items)
        ),
        "correct": correct,
    }


def multi_choice_metrics(items):
    targets = [choice_labels(row["target"]) for row in items]
    predictions = [choice_labels(row["prediction"]) for row in items]
    label_universe = sorted(set().union(*targets, *predictions))
    total_tp = total_fp = total_fn = 0
    per_label_f1 = {}
    sample_jaccard = []
    sample_f1 = []
    exact = 0
    for prediction, target in zip(predictions, targets):
        intersection = len(prediction & target)
        union = len(prediction | target)
        sample_jaccard.append(safe_div(intersection, union) if union else 1.0)
        sample_f1.append(safe_div(2 * intersection, len(prediction) + len(target)))
        exact += prediction == target
        total_tp += intersection
        total_fp += len(prediction - target)
        total_fn += len(target - prediction)
    for label in label_universe:
        tp = sum(label in pred and label in target for pred, target in zip(predictions, targets))
        fp = sum(label in pred and label not in target for pred, target in zip(predictions, targets))
        fn = sum(label not in pred and label in target for pred, target in zip(predictions, targets))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        per_label_f1[label] = safe_div(2 * precision * recall, precision + recall)
    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_tp + total_fn)
    return {
        "records": len(items),
        "label_universe": label_universe,
        "exact_set_accuracy": safe_div(exact, len(items)),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": safe_div(
            2 * micro_precision * micro_recall, micro_precision + micro_recall
        ),
        "macro_f1": safe_div(sum(per_label_f1.values()), len(per_label_f1)),
        "sample_f1": safe_div(sum(sample_f1), len(sample_f1)),
        "jaccard_index": safe_div(sum(sample_jaccard), len(sample_jaccard)),
        "per_label_f1": per_label_f1,
    }


def open_ended_metrics(
    items,
    compute_bertscore=False,
    bertscore_model="bert-base-multilingual-cased",
    bertscore_batch_size=16,
    bertscore_device=None,
):
    predictions = [answer_text(row["prediction"]) for row in items]
    targets = [answer_text(row["target"]) for row in items]
    rouge_scores = [
        rouge_l_f1(prediction, target) for prediction, target in zip(predictions, targets)
    ]
    cider = cider_scores(predictions, targets)
    result = {
        "records": len(items),
        "rouge_l_f1": safe_div(sum(rouge_scores), len(rouge_scores)),
        "cider": safe_div(sum(cider), len(cider)),
        "cider_note": "Corpus-IDF CIDEr, one reference per sample, 0-10 scale",
        "bertscore_model": bertscore_model,
        "bertscore_computed": False,
    }
    if compute_bertscore:
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise RuntimeError(
                "BERTScore requested but bert-score is not installed. "
                "Run: python -m pip install bert-score==0.3.13"
            ) from exc
        device = bertscore_device or ("cuda" if torch.cuda.is_available() else "cpu")
        scorer = BERTScorer(
            model_type=bertscore_model,
            lang="vi",
            device=device,
            rescale_with_baseline=False,
        )
        precision, recall, f1 = scorer.score(
            predictions, targets, batch_size=bertscore_batch_size
        )
        result.update(
            {
                "bertscore_computed": True,
                "bertscore_precision": precision.mean().item(),
                "bertscore_recall": recall.mean().item(),
                "bertscore_f1": f1.mean().item(),
                "bertscore_hash": scorer.hash,
                "bertscore_device": device,
            }
        )
    return result


def aggregate(
    rows,
    compute_bertscore=False,
    bertscore_model="bert-base-multilingual-cased",
    bertscore_batch_size=16,
    bertscore_device=None,
):
    def summarize(items):
        result = {"records": len(items)}
        for metric in ("answer_exact", "answer_token_f1", "location_iou", "location_iou_50"):
            values = [x["metrics"][metric] for x in items if x["metrics"][metric] is not None]
            result[metric] = sum(values) / len(values) if values else None
            result[f"{metric}_count"] = len(values)
        return result

    by_type = defaultdict(list)
    for row in rows:
        by_type[row.get("q_type", "unknown")].append(row)
    task_metrics = {}
    if by_type["closed_ended_questions"]:
        task_metrics["closed_ended_questions"] = closed_ended_metrics(
            by_type["closed_ended_questions"]
        )
    if by_type["single_choice_questions"]:
        task_metrics["single_choice_questions"] = single_choice_metrics(
            by_type["single_choice_questions"]
        )
    if by_type["multi_choice_questions"]:
        task_metrics["multi_choice_questions"] = multi_choice_metrics(
            by_type["multi_choice_questions"]
        )
    if by_type["open_ended_questions"]:
        task_metrics["open_ended_questions"] = open_ended_metrics(
            by_type["open_ended_questions"],
            compute_bertscore=compute_bertscore,
            bertscore_model=bertscore_model,
            bertscore_batch_size=bertscore_batch_size,
            bertscore_device=bertscore_device,
        )
    return {
        "overall": summarize(rows),
        "by_q_type": {key: summarize(value) for key, value in sorted(by_type.items())},
        "task_metrics": task_metrics,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="BoKelvin/GEMeX-VQA-Model-Simple")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=root / "data" / "bronchoscopy_test_15.json"
    )
    parser.add_argument("--image-folder", type=Path, default=root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Recompute metrics from an existing prediction JSONL without loading GEMeX.",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Compute multilingual BERTScore for open-ended answers.",
    )
    parser.add_argument(
        "--bertscore-model", default="bert-base-multilingual-cased"
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument(
        "--bertscore-device",
        choices=("cuda", "cpu"),
        default=None,
    )
    args = parser.parse_args()

    with args.data.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    processed = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing.append(row)
                    processed.add(row["index"])

    pending = [index for index in range(len(dataset)) if index not in processed]
    if args.max_samples is not None:
        pending = pending[: args.max_samples]
    print(
        f"Dataset={len(dataset)}, existing={len(existing)}, "
        f"pending_this_run={len(pending)}",
        flush=True,
    )
    if args.metrics_only or not pending:
        if not existing:
            raise RuntimeError("No existing predictions found for metrics-only evaluation")
        metrics = aggregate(
            existing,
            compute_bertscore=args.bertscore,
            bertscore_model=args.bertscore_model,
            bertscore_batch_size=args.bertscore_batch_size,
            bertscore_device=args.bertscore_device,
        )
        metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Metrics: {metrics_path}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit evaluation")
    device = f"cuda:{args.device}"
    model, tokenizer, image_processor, image_token_len = load_model(
        args.base_model, str(args.adapter), args.device
    )

    with args.output.open("a", encoding="utf-8") as output_handle:
        for done, index in enumerate(pending, start=1):
            item = dataset[index]
            image_path = args.image_folder / item["image"]
            image = Image.open(image_path).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ].to(device=device, dtype=torch.float16)
            prompt = format_prompt(item, image_token_len)
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prediction = tokenizer.decode(
                generated[0, input_ids.shape[1] :], skip_special_tokens=True
            )
            prediction = prediction.split("###", 1)[0].strip()
            target = next(x["value"] for x in item["conversations"] if x["from"] == "gpt")
            row = {
                "index": index,
                "row_id": item.get("row_id"),
                "case_id": item.get("case_id"),
                "q_type": item.get("q_type"),
                "question_type": item.get("question_type"),
                "image": item["image"],
                "question": next(
                    x["value"] for x in item["conversations"] if x["from"] == "human"
                ).replace(DEFAULT_IMAGE_TOKEN, "").strip(),
                "target": target,
                "prediction": prediction,
                "metrics": score_prediction(prediction, target),
            }
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()
            existing.append(row)
            print(
                f"[{done}/{len(pending)}] index={index} "
                f"exact={row['metrics']['answer_exact']:.0f} "
                f"f1={row['metrics']['answer_token_f1']:.3f}",
                flush=True,
            )

    del model
    torch.cuda.empty_cache()
    metrics = aggregate(
        existing,
        compute_bertscore=args.bertscore,
        bertscore_model=args.bertscore_model,
        bertscore_batch_size=args.bertscore_batch_size,
        bertscore_device=args.bertscore_device,
    )
    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {args.output}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
