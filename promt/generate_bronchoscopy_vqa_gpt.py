import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
import requests


MODEL = "gen-VQA"
DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"

# ---------------------------------------------------------------------------
# System prompt for VQA generation – matched exactly from prompt.txt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Bạn là trợ lý AI chuyên tạo bộ câu hỏi - trả lời VQA cho ảnh nội soi phế quản.

Bạn được cung cấp một ảnh nội soi phế quản dưới dạng các mô tả thị giác nguyên tử, mỗi mô tả có thể đi kèm vùng ảnh tương ứng.

Nhiệm vụ:
Tạo các câu hỏi - trả lời về nội dung nhìn thấy trên ảnh nội soi phế quản.

Ngôn ngữ đầu ra:
- Toàn bộ question, answer, reason, options, note phải viết bằng tiếng Việt.
- Không trộn tiếng Anh và tiếng Việt.
- Giữ nguyên các token vùng ảnh dạng <loc_...> trong trường visual_clue.
- Các tên trường JSON giữ nguyên bằng tiếng Anh.

Số lượng mong muốn nếu dữ liệu đủ mạnh:
- 3 câu hỏi open-ended
- 2 câu hỏi closed-ended
- 3 câu hỏi single-choice
- 3 câu hỏi multi-choice

Tuy nhiên, không được ép sinh đủ số lượng nếu dữ liệu thị giác không đủ tin cậy.

Các trường đầu vào có thể gồm:
- case_id
- object_id
- image_path
- report
- sentence_id
- text
- loc_token
- anatomical_location
- finding
- attribute_type
- color_label
- color_text

Các loại attribute_type có thể gặp:
- mucosa_status: tình trạng niêm mạc, ví dụ niêm mạc thâm nhiễm, tăng sinh mạch, xung huyết
- secretion_status: dịch tiết, dịch mủ
- airway_lumen_status: tình trạng lòng khí phế quản, tổn thương trong lòng khí phế quản
- stenosis_degree: mức độ hẹp lòng phế quản
- region_shape_size: kích thước, phạm vi, hình dạng vùng đánh dấu
- anatomical_location: vị trí giải phẫu nếu có

Các loại câu hỏi hợp lệ:
- mucosa_status
- secretion_status
- airway_lumen_status
- stenosis_degree
- abnormality
- severity
- clinical_implication
- anatomical_location
- region_shape_size
- visual_description

Nguyên tắc ưu tiên nội dung:

(1) Ưu tiên cao nhất cho các câu hỏi có giá trị y khoa trực tiếp:
- mucosa_status
- secretion_status
- airway_lumen_status
- stenosis_degree
- abnormality
- severity nếu được nêu rõ
- clinical_implication nếu có thể suy ra rất thận trọng từ finding

(2) Chỉ dùng anatomical_location và region_shape_size như thông tin hỗ trợ, không để chúng chiếm phần lớn bộ câu hỏi.

(3) Nếu report có ít nhất một finding y khoa thuộc các nhóm:
- mucosa_status
- secretion_status
- airway_lumen_status
- stenosis_degree

thì tổng số câu hỏi thuộc các nhóm anatomical_location, region_shape_size, visual_description không được vượt quá 3 câu trong toàn bộ output.

(4) Nếu report chỉ có region_shape_size và không có finding y khoa thật, không sinh đủ 11 câu. Chỉ sinh tối đa:
- 1 open-ended
- 1 closed-ended
- 1 single-choice
- 1 multi-choice

và note phải là:
"Chỉ có thông tin về vị trí/hình dạng vùng đánh dấu, nên số lượng câu hỏi được giới hạn để tránh suy diễn y khoa."

(5) Nếu report rỗng, trả về đúng JSON sau:

{
  "case_id": "...",
  "object_id": "...",
  "open_ended_questions": [],
  "closed_ended_questions": [],
  "single_choice_questions": [],
  "multi_choice_questions": [],
  "note": "Không có đủ thông tin thị giác đáng tin cậy để tạo câu hỏi."
}

Quy tắc về câu hỏi:

(6) Câu hỏi phải hỏi về nội dung nhìn thấy trên ảnh nội soi phế quản.

(7) Không được hỏi những thông tin không thể trả lời chắc chắn từ ảnh hoặc finding đã cho, ví dụ:
- triệu chứng của bệnh nhân
- tiền sử bệnh
- kết quả mô bệnh học
- chẩn đoán xác định
- nguyên nhân bệnh
- tiên lượng
- điều trị
- chỉ định can thiệp
- so sánh với lần nội soi trước
- diễn tiến theo thời gian
- kết quả xét nghiệm khác
- CT, X-quang, MRI hoặc phương tiện hình ảnh khác

(8) Không được nhắc các từ sau trong question, answer hoặc reason:
- report
- annotation
- mô tả
- phrase
- text
- loc_token
- provided information
- dữ liệu được cung cấp
- ghi nhận trong báo cáo
- theo báo cáo
- finding states
- explicitly states
- atomic finding
- documented

(9) Không đưa token <loc_...> vào câu hỏi hoặc câu trả lời. Token vùng ảnh chỉ được xuất hiện trong visual_clue.

Sai:
"Có dịch mủ ở vùng <loc_59_34_533_451> không?"

Đúng:
"Có dịch mủ trong vùng được đánh dấu không?"

(10) Không dùng các từ:
- chính
- chủ đạo
- nổi bật nhất
- nghiêm trọng nhất
- lớn nhất
- nhỏ nhất
- dominant
- main
- primary
- most severe
- largest
- smallest

trừ khi input nêu rõ sự so sánh đó.

(11) Không tự tạo vị trí giải phẫu. Nếu anatomical_location là null hoặc chuỗi rỗng, đặt anatomical_location là null trong visual_clue và không hỏi câu hỏi "ở vị trí giải phẫu nào".

(12) Không tự suy luận màu sắc. Chỉ hỏi về màu nếu color_label hoặc color_text khác null.

(13) Không tự suy luận bệnh danh. Không kết luận ung thư, lao, viêm phổi, u ác tính, nhiễm trùng hoặc bệnh cụ thể nếu finding không nêu rõ.

(14) Với clinical_implication, chỉ được dùng ngôn ngữ thận trọng:
- "gợi ý bất thường niêm mạc"
- "có thể liên quan đến thay đổi viêm hoặc thâm nhiễm"
- "gợi ý cản trở một phần lòng phế quản"
- "gợi ý có dịch tiết bất thường"

Không được dùng:
- "chẩn đoán"
- "xác định"
- "chắc chắn"
- "ác tính"
- "ung thư"
- "lao"
- "viêm phổi"

trừ khi input nêu trực tiếp.

Quy tắc về reason:

(15) Reason phải giải thích dựa trên dấu hiệu nhìn thấy, không được viết như đang đọc report.

Sai:
"Finding states niêm mạc thâm nhiễm."

Đúng:
"Vùng được đánh dấu có biểu hiện niêm mạc bất thường dạng thâm nhiễm."

Sai:
"The region is described as large and irregular."

Đúng:
"Vùng được đánh dấu có phạm vi rộng và phân bố không đều."

(16) Reason phải nhất quán ngôn ngữ tiếng Việt.

Quy tắc về loại câu hỏi:

(17) Open-ended:
- Câu trả lời phải ngắn gọn.
- Ưu tiên cụm y khoa ngắn.
Ví dụ:
"Dịch mủ."
"Niêm mạc thâm nhiễm và tăng sinh mạch."
"Hẹp lòng phế quản 26–50%."

(18) Closed-ended:
- answer chỉ được là "yes" hoặc "no".
- Câu hỏi phải rõ ràng và có thể kiểm chứng bằng visual_clue.

(19) Single-choice:
- Có đúng 4 lựa chọn A, B, C, D.
- Chỉ có 1 đáp án đúng.
- answer phải gồm cả chữ cái và nội dung.
Ví dụ:
"B. Tăng sinh mạch"

(20) Multi-choice:
- Có 4 hoặc 5 lựa chọn.
- Có từ 2 đáp án đúng trở lên nếu có thể.
- answer là list.
Ví dụ:
["A. Tăng sinh mạch", "C. Niêm mạc thâm nhiễm"]

(21) Các lựa chọn sai phải hợp lý trong nội soi phế quản nhưng không được trái với finding đã cho.

Quy tắc về đa dạng câu hỏi:

(22) Nếu có nhiều loại finding y khoa, phải phân bố câu hỏi tương đối đều.
Không được tạo quá nhiều câu hỏi chỉ xoay quanh hình dạng, kích thước hoặc vị trí.

(23) Với ảnh có các finding như dịch mủ, tăng sinh mạch, niêm mạc thâm nhiễm, hẹp lòng phế quản, tổn thương trong lòng khí phế quản:
- phải ưu tiên hỏi về các finding này trước.
- region_shape_size chỉ được dùng tối đa 2 câu.
- anatomical_location chỉ được dùng tối đa 1 câu.

(24) Nếu nhiều finding cùng chung một loc_token, có thể tạo câu hỏi multi-choice để hỏi các bất thường cùng xuất hiện trong cùng vùng đó.

(25) Nếu câu hỏi liên quan đến nhiều vùng ảnh, visual_clue phải chứa tất cả các vùng liên quan.

Quy tắc visual_clue:

(26) visual_clue luôn là list, kể cả khi chỉ có một vùng.

Ví dụ:
"visual_clue": [
  {
    "loc_token": "<loc_153_28_642_474>",
    "anatomical_location": null
  }
]

(27) Mỗi phần tử trong visual_clue phải có:
- loc_token
- anatomical_location

Nếu anatomical_location rỗng hoặc null, dùng null.

Schema đầu ra bắt buộc:

{
  "case_id": "...",
  "object_id": "...",
  "open_ended_questions": [
    {
      "question": "...",
      "question_format": "open-ended",
      "question_type": "...",
      "answer": "...",
      "reason": "...",
      "visual_clue": [
        {
          "loc_token": "...",
          "anatomical_location": null
        }
      ]
    }
  ],
  "closed_ended_questions": [
    {
      "question": "...",
      "question_format": "closed-ended",
      "question_type": "...",
      "answer": "yes",
      "reason": "...",
      "visual_clue": [
        {
          "loc_token": "...",
          "anatomical_location": null
        }
      ]
    }
  ],
  "single_choice_questions": [
    {
      "question": "...",
      "question_format": "single-choice",
      "question_type": "...",
      "options": {
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      },
      "answer": "B. ...",
      "reason": "...",
      "visual_clue": [
        {
          "loc_token": "...",
          "anatomical_location": null
        }
      ]
    }
  ],
  "multi_choice_questions": [
    {
      "question": "...",
      "question_format": "multi-choice",
      "question_type": "...",
      "options": {
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "...",
        "E": "..."
      },
      "answer": ["A. ...", "C. ..."],
      "reason": "...",
      "visual_clue": [
        {
          "loc_token": "...",
          "anatomical_location": null
        }
      ]
    }
  ],
  "note": null
}

Output chỉ được là JSON hợp lệ. Không thêm Markdown, giải thích hoặc văn bản ngoài JSON.
"""


# ---------------------------------------------------------------------------
# Helper utilities (mirror generate_bronchoscopy_reports_gpt.py)
# ---------------------------------------------------------------------------


def _extract_content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _extract_first_json_object(text):
    content = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if fenced:
        content = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            parsed, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Model output is not valid JSON object: {text[:500]}")


def _decode_response_text(response):
    raw = response.content
    if not raw:
        return response.text
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return response.text


def _load_response_json(response):
    decoded_text = _decode_response_text(response)
    try:
        parsed = json.loads(decoded_text)
    except json.JSONDecodeError:
        parsed = _extract_first_json_object(decoded_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid JSON response type: {type(parsed)} | Raw: {decoded_text[:1000]}")
    return parsed


# ---------------------------------------------------------------------------
# Build messages from report – matched exactly from prompt.txt template
# ---------------------------------------------------------------------------


def build_messages(case_id, object_id, image_path, report):
    """Build the OpenAI-format messages list for VQA generation."""
    user_content = f"""Ảnh nội soi phế quản:

case_id: {case_id}
object_id: {object_id}
image_path: {image_path}

Các finding thị giác nguyên tử:
{json.dumps(report, ensure_ascii=False, indent=2)}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(api_key, base_url, case_id, object_id, image_path, report, model,
             max_retries=3, base_sleep=2.0, max_tokens=12000):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = build_messages(case_id, object_id, image_path, report)
    payload = {
        "model": model,
        "messages": messages,
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
                print(
                    f"[WARN] Request failed: {exc}. Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s.",
                    file=sys.stderr,
                )
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
            raise requests.HTTPError(
                f"API error {response.status_code}: {response.text[:1000]}",
                response=response,
            )

        response_json = _load_response_json(response)
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"No choices returned by API. Raw response: {response_json}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError(f"Missing assistant message from API. Raw response: {response_json}")
        text = _extract_content_text(message.get("content"))
        if not text:
            reasoning_text = _extract_content_text(message.get("reasoning_content"))
            if reasoning_text and attempt < max_retries:
                wait_time = base_sleep * (attempt + 1)
                print(
                    f"[WARN] Empty content, reasoning-only response. Retry {attempt + 1}/{max_retries} after {wait_time:.1f}s.",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                attempt += 1
                continue
            raise ValueError(f"API response has empty text. Raw response: {response_json}")
        vqa = _extract_first_json_object(text)
        return vqa


# ---------------------------------------------------------------------------
# Validate VQA output
# ---------------------------------------------------------------------------


def validate_vqa(vqa, case_id, object_id, image_path):
    """Ensure the VQA output has required top-level keys and correct structure."""
    required_keys = ["open_ended_questions", "closed_ended_questions",
                     "single_choice_questions", "multi_choice_questions"]
    for key in required_keys:
        if key not in vqa:
            vqa[key] = []

    # Override case_id/object_id from input data to ensure consistency
    vqa["case_id"] = case_id
    vqa["object_id"] = object_id
    vqa["image_path"] = image_path

    # Ensure note field exists
    if "note" not in vqa:
        vqa["note"] = None

    return vqa


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------


def output_path_for(input_path, output_root=None):
    """Derive _vqa.json path from the _reports.json input path."""
    input_path = Path(input_path)
    name = input_path.name
    if name.endswith("_reports.json"):
        name = name[:-13] + "_vqa.json"
    elif name.endswith(".json"):
        name = name[:-5] + "_vqa.json"
    else:
        name = name + "_vqa.json"

    if output_root:
        rel = input_path.parent.relative_to(
            Path("/home/ailab/Documents/KC4.0_Final_annots_data_png/Cabenh_reports")
        )
        out_dir = Path(output_root) / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / name
    return input_path.with_name(name)


def load_existing_vqa(out_path):
    """Load previously saved VQA entries (list) if any."""
    if not out_path.exists():
        return []
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Existing VQA file must be a list: {out_path}")
    return data


def save_vqa(out_path, vqa_list):
    """Atomically save the VQA list to disk."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(vqa_list, f, ensure_ascii=False, indent=2)
    tmp_path.replace(out_path)


# ---------------------------------------------------------------------------
# Process a single _reports.json file
# ---------------------------------------------------------------------------


def process_file(api_key, base_url, json_path, model, output_root=None,
                 overwrite=False, max_retries=5, base_sleep=5.0, max_tokens=12000):
    json_path = Path(json_path)
    out_path = output_path_for(json_path, output_root)

    with open(json_path, "r", encoding="utf-8") as f:
        reports_data = json.load(f)

    vqa_list = [] if overwrite else load_existing_vqa(out_path)
    done_object_ids = {r.get("object_id") for r in vqa_list if isinstance(r, dict)}

    for idx, item in enumerate(reports_data, start=1):
        object_id = item.get("object_id")
        image_path = item.get("image_path", "")
        case_id = item.get("case_id", "")
        report = item.get("report", [])

        if object_id in done_object_ids:
            print(f"  {idx}/{len(reports_data)} {object_id} SKIP")
            continue

        vqa = call_llm(
            api_key, base_url, case_id, object_id, image_path, report, model,
            max_retries=max_retries, base_sleep=base_sleep, max_tokens=max_tokens,
        )
        vqa = validate_vqa(vqa, case_id, object_id, image_path)
        vqa_list.append(vqa)
        done_object_ids.add(object_id)
        save_vqa(out_path, vqa_list)
        print(f"  {idx}/{len(reports_data)} {object_id} SAVED")

    print(f"SAVED {len(vqa_list)} VQA entries -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def iter_inputs(path):
    """Yield all _reports.json files under the given path."""
    path = Path(path)
    if path.is_file():
        yield path
        return
    yield from sorted(path.rglob("*_reports.json"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate VQA questions from bronchoscopy reports using LLM."
    )
    parser.add_argument(
        "input",
        help="A _reports.json file or a Cabenh_reports folder with *_reports.json files",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NINEROUTER_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--api-key-env", default="NINEROUTER_API_KEY")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional root folder for VQA output",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--base-sleep", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=12000)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing {args.api_key_env} or OPENAI_API_KEY")

    files = list(iter_inputs(args.input))
    if args.limit:
        files = files[: args.limit]

    print(f"Input files: {len(files)}")
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path}")
        process_file(
            api_key,
            args.base_url,
            path,
            args.model,
            args.output_root,
            args.overwrite,
            args.max_retries,
            args.base_sleep,
            args.max_tokens,
        )


if __name__ == "__main__":
    main()
