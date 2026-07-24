import argparse
import json
import re
from pathlib import Path


Q_TYPE_PROMPTS = {
    "open_ended_questions": (
        "Input an open-ended question, and the assistant will output its "
        "answer with a detailed reason and corresponding visual location."
    ),
    "closed_ended_questions": (
        "Input a closed-ended question, and the assistant will output its "
        "answer (yes or no) with a detailed reason and corresponding visual "
        "location."
    ),
    "single_choice_questions": (
        "Input a single-choice question, and the assistant will output its "
        "answer (an option) with a detailed reason and corresponding visual "
        "location."
    ),
    "multi_choice_questions": (
        "Input a multi-choice question, and the assistant will output its "
        "answer (some options) with a detailed reason and corresponding "
        "visual location."
    ),
}

FORMAT_TO_GROUP = {
    "open-ended": "open_ended_questions",
    "closed-ended": "closed_ended_questions",
    "single-choice": "single_choice_questions",
    "multi-choice": "multi_choice_questions",
}

LOC_RE = re.compile(r"<loc_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)>")


def load_json_or_jsonl(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        loaded = json.loads(text)
        return loaded if isinstance(loaded, list) else [loaded]
    if text[0] == "{":
        try:
            loaded = json.loads(text)
            return loaded if isinstance(loaded, list) else [loaded]
        except json.JSONDecodeError:
            pass

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def answer_to_text(answer):
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        return ", ".join(str(item) for item in answer)
    return json.dumps(answer, ensure_ascii=False)


def options_to_choices(options):
    if not options:
        return []
    if isinstance(options, dict):
        return [f"{key}: {value}" for key, value in options.items()]
    if isinstance(options, list):
        return [str(item) for item in options]
    return [str(options)]


def loc_tokens_to_boxes(item):
    boxes = []
    for clue in item.get("visual_clue") or []:
        if not isinstance(clue, dict):
            continue
        loc_token = clue.get("loc_token")
        if not loc_token:
            continue
        match = LOC_RE.fullmatch(loc_token)
        if match:
            boxes.append([int(value) for value in match.groups()])

    if boxes:
        return boxes

    for loc_token in item.get("visual_regions") or []:
        match = LOC_RE.fullmatch(str(loc_token))
        if match:
            boxes.append([int(value) for value in match.groups()])
    return boxes


def question_group(item):
    group = item.get("question_group")
    if group in Q_TYPE_PROMPTS:
        return group
    return FORMAT_TO_GROUP.get(item.get("question_format"), "open_ended_questions")


def convert_item(item, row_id):
    group = question_group(item)
    choices = options_to_choices(item.get("options"))
    human_value = f"<image>\n{item.get('question', '')}"
    if choices:
        human_value += " <choices>: [" + ", ".join(choices) + "]"

    locations = loc_tokens_to_boxes(item)
    gpt_value = (
        f"<answer> {answer_to_text(item.get('answer'))} "
        f"<reason> {item.get('reason') or ''} "
        f"<location> {locations}"
    )

    return {
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value},
        ],
        "row_id": row_id,
        "case_id": item.get("case_id", ""),
        "object_id": item.get("object_id", ""),
        "q_type": group,
        "question_type": item.get("question_type") or item.get("type", ""),
        "type_prompt": Q_TYPE_PROMPTS[group],
        "image": item.get("image_path", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/ailab/Documents/KC4.0_Final_annots_data_png/all_vqa.json",
        help="Raw all_vqa JSON/JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/ailab/Documents/KC4.0_Final_annots_data_png/GEMeX-Project/"
            "data-v1-subset/bronchoscopy_train_data.json"
        ),
        help="Output instruction JSON file for llava/train/train_mem.py.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    raw_items = load_json_or_jsonl(input_path)
    converted = [convert_item(item, idx) for idx, item in enumerate(raw_items)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    missing_images = sum(1 for item in converted if not Path(item["image"]).exists())
    print(f"Input records: {len(raw_items)}")
    print(f"Output records: {len(converted)}")
    print(f"Missing image paths: {missing_images}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
