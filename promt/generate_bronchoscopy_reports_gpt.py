#!/usr/bin/env python3
"""
Generate one bronchoscopy report per image from llm_input JSON and image files.

The output is a list of image-level reports. That list can be used later as
input for a case-level summarization step.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from PIL import Image


MODEL = "gen-VQA"
DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"
DEFAULT_CABENH_DIR = Path("/home/ailab/Documents/KC4.0_Final_annots_data_png/Cabenh")
REPORT_TYPE = "single_image_natural_atomic_report"


SYSTEM_PROMPT = """Bạn là hệ thống tạo báo cáo mô tả MỘT ảnh nội soi phế quản từ ảnh và annotation có cấu trúc.

Mục tiêu:
Tạo một report ngắn, tự nhiên, khách quan cho đúng một ảnh nội soi phế quản. Report này là đơn vị trung gian; các report theo ảnh sẽ được gom lại ở bước sau để tạo report lớn hơn cho toàn ca bệnh.

Nguyên tắc bắt buộc:
1. Mỗi object_id tương ứng với một report riêng.
2. Chỉ mô tả ảnh/object_id đang được cung cấp. Không viết kết luận chung cho cả ca bệnh.
3. Dùng annotation làm nguồn sự thật chính cho finding, vị trí và loc_token.
4. Có thể dùng ảnh để giúp câu mô tả tự nhiên hơn, nhưng không thêm finding mới ngoài gt_finding và finding_labels.
5. Không tự suy luận chẩn đoán bệnh, nguyên nhân, giai đoạn, mức độ ác tính hoặc tiên lượng nếu annotation không cung cấp.
6. Nếu gt_location bị trùng hoàn toàn về location_labels, finding_labels, bbox và loc_token thì chỉ mô tả một lần.
7. Nếu finding nằm trong gt_location[i].finding_labels thì phải gắn với đúng loc_token và location_labels của gt_location[i].
8. Nếu finding có trong gt_finding nhưng không nằm trong bất kỳ finding_labels nào, hãy tạo câu mô tả ở mức ảnh, không tự gán bbox.
9. Nếu location_labels rỗng, không được tự suy luận mốc giải phẫu; chỉ dùng loc_token hoặc object_id để truy vết nếu cần.
10. Nếu gt_location[i] có color_analysis hợp lệ:
    - color_analysis.color_label khác null/rỗng
    - color_analysis.error = null
    - color_analysis.valid_ratio >= 0.8
    thì có thể tạo câu riêng với attribute_type = "color_observation".
11. Màu sắc từ color_analysis là màu của vùng quan sát/ROI, không nhất thiết là màu của finding. Viết an toàn:
    - “vùng quan sát có xu hướng đỏ”
    - “vùng quan sát có xu hướng vàng”
    - “vùng quan sát có xu hướng trắng”
    - “vùng quan sát nhợt màu”
    - “vùng quan sát sẫm màu”
12. Không viết “dịch mủ màu đỏ”, “tổn thương màu đỏ”, “khối màu đỏ” trừ khi annotation trực tiếp nói như vậy.
13. Với finding “Niêm mạc xung huyết”, mô tả là tình trạng niêm mạc xung huyết.
14. Với finding “Hẹp lòng phế quản” và các finding mức độ hẹp như “Hẹp từ 26 đến 50%”, mô tả thành tình trạng hẹp và mức độ hẹp.
15. Với finding như “Dịch mủ”, “dịch nhầy”, “máu”, “đờm”, xếp vào secretion_status hoặc airway_content, không xếp vào airway_lumen_status.
16. Nếu có area_text và shape, có thể thêm một câu riêng mô tả vùng đánh dấu, nhưng phải viết an toàn:
    “Vùng đánh dấu trên ảnh có phạm vi nhỏ và dạng tương đối tròn.”
    Không viết như thể đó chắc chắn là một khối bệnh lý.
17. Nếu ảnh không có gt_finding và không có gt_location chứa finding_labels, hãy viết ngắn rằng annotation của ảnh này không ghi nhận bất thường cụ thể; không khẳng định ảnh bình thường nếu annotation không có nhãn bình thường.
18. Văn phong cần tự nhiên, ngắn gọn, khách quan, giống mô tả y khoa đơn giản.
19. Output phải là JSON hợp lệ, không markdown, không giải thích.

Cách viết câu:
- Không viết máy móc kiểu: “Ở vị trí bbox là...”.
- Ưu tiên viết tự nhiên:
  “Tại vùng phế quản phân thuỳ lưỡi trái, tương ứng vị trí <loc_...>, ghi nhận...”
  “Tại vị trí <loc_...>, ghi nhận...”
  “Niêm mạc tại vùng quan sát này có biểu hiện...”
  “Vùng quan sát tại vị trí này có xu hướng...”
  “Mức độ hẹp tại vị trí này được ghi nhận...”
  “Vùng đánh dấu trên ảnh có phạm vi..., dạng...”

Format output bắt buộc:
{
  "case_id": "...",
  "object_id": "...",
  "image_path": "...",
  "report_type": "single_image_natural_atomic_report",
  "report": [
    {
      "sentence_id": 1,
      "text": "...",
      "loc_token": "...",
      "anatomical_location": "...",
      "finding": "...",
      "attribute_type": "...",
      "color_label": null,
      "color_text": null
    }
  ],
  "final_report": "..."
}

attribute_type hợp lệ:
- "airway_lumen_status": tình trạng lòng phế quản, ví dụ hẹp lòng phế quản, tắc lòng phế quản
- "stenosis_degree": mức độ hẹp, ví dụ hẹp từ 26 đến 50%
- "mucosa_status": tình trạng niêm mạc, ví dụ niêm mạc xung huyết, niêm mạc thâm nhiễm, phù nề
- "secretion_status": dịch tiết, ví dụ dịch mủ, dịch nhầy, máu, đờm
- "airway_content": nội dung trong lòng phế quản khi không phù hợp secretion_status
- "color_observation": màu sắc vùng quan sát
- "region_shape_size": kích thước/hình dạng vùng đánh dấu
- "image_level_finding": finding không có bbox cụ thể

Quy tắc color_label và color_text:
- Với câu không phải color_observation: color_label = null, color_text = null.
- Với câu color_observation:
  + color_label lấy trực tiếp từ color_analysis.color_label.
  + color_text là cụm mô tả màu an toàn, ví dụ “vùng quan sát có xu hướng đỏ”.
"""


def dedupe_keep_order(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        out.append(text)
        seen.add(key)
    return out


def compact_location(loc: Dict[str, Any]) -> Dict[str, Any]:
    color = loc.get("color_analysis") or {}
    color_compact = None
    if color:
        color_compact = {
            "color_label": color.get("color_label"),
            "mean_rgb": color.get("mean_rgb"),
            "median_rgb": color.get("median_rgb"),
            "valid_ratio": color.get("valid_ratio"),
            "error": color.get("error"),
        }

    return {
        "location_labels": loc.get("location_labels", []),
        "finding_labels": loc.get("finding_labels", []),
        "other_labels": loc.get("other_labels", []),
        "bbox": loc.get("bbox") or {},
        "loc_token": loc.get("loc_token"),
        "area_text": loc.get("area_text"),
        "shape": loc.get("shape"),
        "color_analysis": color_compact,
    }


def compact_annotation(item: Dict[str, Any]) -> Dict[str, Any]:
    kept = {
        "case_id": item.get("case_id", ""),
        "object_id": item.get("object_id", ""),
        "image_path": item.get("image_path", ""),
        "image_width": item.get("image_width"),
        "image_height": item.get("image_height"),
        "gt_finding": item.get("gt_finding", []),
        "gt_location": [],
    }

    seen = set()
    for loc in item.get("gt_location", []) or []:
        compact = compact_location(loc)
        key = json.dumps(
            {
                "location_labels": compact["location_labels"],
                "finding_labels": compact["finding_labels"],
                "bbox": compact["bbox"],
                "loc_token": compact["loc_token"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        kept["gt_location"].append(compact)
    return kept


def encode_image_data_url(image_path: str, max_side: int = 768, quality: int = 85) -> Optional[str]:
    path = Path(image_path)
    if not image_path or not path.exists():
        return None

    with Image.open(path) as img:
        img = img.convert("RGB")
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_message_content(annotation: Dict[str, Any], use_images: bool, image_max_side: int, image_detail: str) -> Any:
    user_text = (
        SYSTEM_PROMPT
        + "\n\nDữ liệu annotation của đúng một ảnh:\n"
        + json.dumps(annotation, ensure_ascii=False, indent=2)
        + "\n\nChỉ trả về JSON hợp lệ, không markdown, không giải thích."
    )
    if not use_images:
        return user_text

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    data_url = encode_image_data_url(annotation.get("image_path", ""), max_side=image_max_side)
    if data_url:
        content.append({"type": "text", "text": f"Ảnh nội soi tương ứng object_id={annotation.get('object_id', '')}"})
        content.append({"type": "image_url", "image_url": {"url": data_url, "detail": image_detail}})
    return content


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    content = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if fenced:
        content = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            parsed, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Model output is not valid JSON object: {text[:500]}")


def _decode_response_text(response: requests.Response) -> str:
    raw = response.content
    if not raw:
        return response.text
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return response.text


def _load_response_json(response: requests.Response) -> Dict[str, Any]:
    decoded_text = _decode_response_text(response)
    try:
        parsed = json.loads(decoded_text)
    except json.JSONDecodeError:
        parsed = _extract_first_json_object(decoded_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid JSON response type: {type(parsed)} | Raw: {decoded_text[:1000]}")
    return parsed


def call_llm(
    api_key: str,
    base_url: str,
    annotation: Dict[str, Any],
    model: str,
    use_images: bool,
    image_max_side: int,
    image_detail: str,
    max_retries: int = 3,
    base_sleep: float = 2.0,
    max_tokens: int = 12000,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_message_content(
                    annotation,
                    use_images=use_images,
                    image_max_side=image_max_side,
                    image_detail=image_detail,
                ),
            }
        ],
        "max_completion_tokens": max_tokens,
        "stream": False,
    }

    attempt = 0
    while True:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
        except requests.RequestException as exc:
            if attempt < max_retries:
                wait_time = base_sleep * (attempt + 1)
                print(f"[WARN] Request failed: {exc}. Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s.", file=sys.stderr)
                time.sleep(wait_time)
                attempt += 1
                continue
            raise

        if response.status_code in (429, 500, 502, 503, 504, 524) and attempt < max_retries:
            wait_time = base_sleep * (attempt + 1)
            print(
                f"[WARN] API {response.status_code}. Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s. "
                f"Body: {response.text[:300]}",
                file=sys.stderr,
            )
            time.sleep(wait_time)
            attempt += 1
            continue

        if not response.ok:
            raise requests.HTTPError(f"API error {response.status_code}: {response.text[:1000]}", response=response)

        response_json = _load_response_json(response)
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"No choices returned. Raw response: {response_json}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError(f"Missing assistant message. Raw response: {response_json}")
        text = _extract_content_text(message.get("content"))
        if not text:
            text = _extract_content_text(message.get("reasoning_content"))
        if not text:
            raise ValueError(f"Response has empty text. Raw response: {response_json}")

        try:
            report = _extract_first_json_object(text)
        except ValueError as exc:
            if attempt < max_retries:
                wait_time = base_sleep * (attempt + 1)
                print(
                    f"[WARN] Model did not return JSON. Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s. "
                    f"Text: {text[:300]}",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                attempt += 1
                continue
            raise exc
        report["report_type"] = REPORT_TYPE
        return report


def validate_report(report: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
    if not report.get("case_id"):
        report["case_id"] = annotation.get("case_id", "")
    if not report.get("object_id"):
        report["object_id"] = annotation.get("object_id", "")
    if not report.get("image_path"):
        report["image_path"] = annotation.get("image_path", "")

    if report.get("case_id") != annotation.get("case_id"):
        raise ValueError(f"case_id mismatch: {report.get('case_id')} != {annotation.get('case_id')}")
    if report.get("object_id") != annotation.get("object_id"):
        raise ValueError(f"object_id mismatch: {report.get('object_id')} != {annotation.get('object_id')}")

    report["image_path"] = annotation.get("image_path", "")
    report["report_type"] = REPORT_TYPE

    sentences = report.get("report")
    if isinstance(sentences, dict):
        if isinstance(sentences.get("sentences"), list):
            sentences = sentences["sentences"]
        elif isinstance(sentences.get("report"), list):
            sentences = sentences["report"]
        else:
            sentences = [sentences]
    elif isinstance(sentences, str):
        sentences = [{"text": sentences}]
    elif sentences is None:
        fallback_text = report.get("final_report") or report.get("text") or report.get("description")
        sentences = [{"text": str(fallback_text)}] if fallback_text else []
    elif not isinstance(sentences, list):
        sentences = [{"text": str(sentences)}]
    report["report"] = sentences

    for idx, sent in enumerate(sentences, start=1):
        if not isinstance(sent, dict):
            sent = {"text": str(sent)}
            sentences[idx - 1] = sent
        sent["sentence_id"] = idx
        sent.setdefault("text", "")
        sent.setdefault("loc_token", None)
        sent.setdefault("anatomical_location", None)
        sent.setdefault("finding", None)
        sent.setdefault("attribute_type", "image_level_finding")
        sent.setdefault("color_label", None)
        sent.setdefault("color_text", None)
        if sent.get("attribute_type") != "color_observation":
            sent["color_label"] = None
            sent["color_text"] = None

    if not isinstance(report.get("final_report"), str) or not report["final_report"].strip():
        report["final_report"] = " ".join(str(s.get("text", "")).strip() for s in sentences if s.get("text")).strip()
    return report


def output_path_for(input_path: Path, output_root: Optional[str] = None) -> Path:
    input_path = Path(input_path)
    stem = input_path.stem
    stem = stem.replace("_with_colors", "")
    name = f"{stem}_reports.json"

    if output_root:
        try:
            rel = input_path.parent.relative_to(DEFAULT_CABENH_DIR)
        except ValueError:
            rel = Path(input_path.parent.name)
        out_dir = Path(output_root) / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / name
    return input_path.with_name(name)


def load_existing_reports(out_path: Path) -> List[Dict[str, Any]]:
    if not out_path.exists():
        return []
    with out_path.open("r", encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        backup_path = out_path.with_suffix(out_path.suffix + f".invalid_{int(time.time())}")
        shutil.copy2(out_path, backup_path)
        print(
            f"[WARN] Cannot parse existing report JSON, backing it up and regenerating this file: "
            f"{out_path} -> {backup_path}. Error: {exc}",
            file=sys.stderr,
        )
        return []
    if not isinstance(data, list):
        raise ValueError(f"Existing report file must be a list: {out_path}")
    return [item for item in data if isinstance(item, dict)]


def save_reports(out_path: Path, reports: List[Dict[str, Any]]) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    tmp_path.replace(out_path)


def process_file(
    api_key: str,
    base_url: str,
    json_path: Path,
    model: str,
    output_root: Optional[str] = None,
    overwrite: bool = False,
    use_images: bool = True,
    image_max_side: int = 768,
    image_detail: str = "low",
    max_retries: int = 5,
    base_sleep: float = 5.0,
    max_tokens: int = 12000,
    dry_run: bool = False,
) -> Path:
    json_path = Path(json_path)
    out_path = output_path_for(json_path, output_root)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input must be a list of image annotations: {json_path}")

    if dry_run:
        print(f"[dry-run] {json_path} -> {out_path} | image_reports={len(data)}, use_images={use_images}")
        return out_path

    reports = [] if overwrite else load_existing_reports(out_path)
    done_object_ids = {report.get("object_id") for report in reports if isinstance(report, dict)}

    for idx, item in enumerate(data, start=1):
        object_id = item.get("object_id")
        if object_id in done_object_ids:
            print(f"  {idx}/{len(data)} {object_id} SKIP")
            continue

        annotation = compact_annotation(item)
        report = call_llm(
            api_key,
            base_url,
            annotation,
            model,
            use_images=use_images,
            image_max_side=image_max_side,
            image_detail=image_detail,
            max_retries=max_retries,
            base_sleep=base_sleep,
            max_tokens=max_tokens,
        )
        report = validate_report(report, annotation)
        reports.append(report)
        done_object_ids.add(object_id)
        save_reports(out_path, reports)
        print(f"  {idx}/{len(data)} {object_id} SAVED")

    print(f"SAVED {len(reports)} image reports -> {out_path}")
    return out_path


def input_score(path: Path) -> int:
    name = path.name
    if "_with_colors" in name:
        return 0
    if name.endswith(".json"):
        return 1
    return 2


def is_report_or_aux_file(path: Path) -> bool:
    name = path.name
    return (
        "_reports" in name
        or "_case_report" in name
        or "_roi_colors" in name
        or name.endswith(".json.bak")
    )


def iter_inputs(path: str) -> List[Path]:
    root = Path(path)
    if root.is_file():
        return [root]

    grouped: Dict[Path, List[Path]] = {}
    for candidate in sorted(root.rglob("llm_input*.json")):
        if is_report_or_aux_file(candidate):
            continue
        grouped.setdefault(candidate.parent, []).append(candidate)

    selected = []
    for _, candidates in sorted(grouped.items(), key=lambda x: str(x[0])):
        selected.append(sorted(candidates, key=input_score)[0])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one natural bronchoscopy report per image using llm_input JSON and images.")
    parser.add_argument("input", help="A llm_input .json file or Cabenh folder")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=os.environ.get("NINEROUTER_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="NINEROUTER_API_KEY")
    parser.add_argument("--output-root", default=None, help="Optional root folder for reports")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of input llm_input files.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--base-sleep", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--no-images", action="store_true", help="Use annotation only; do not send images to the model.")
    parser.add_argument("--image-max-side", type=int, default=768, help="Resize image so the longest side is at most this value before sending.")
    parser.add_argument("--image-detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        raise RuntimeError(f"Missing {args.api_key_env} or OPENAI_API_KEY")

    files = iter_inputs(args.input)
    if args.limit:
        files = files[: args.limit]

    print(f"Input files: {len(files)}")
    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {path}")
        process_file(
            api_key,
            args.base_url,
            path,
            args.model,
            output_root=args.output_root,
            overwrite=args.overwrite,
            use_images=not args.no_images,
            image_max_side=args.image_max_side,
            image_detail=args.image_detail,
            max_retries=args.max_retries,
            base_sleep=args.base_sleep,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
