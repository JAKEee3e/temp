from __future__ import annotations

import json
import re
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer


BBox = Tuple[float, float, float, float]


@dataclass
class QwenBackend:
    tokenizer: Any
    model: Any


class DialogueLine(BaseModel):
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    tone: Optional[str] = None
    bubble_style: Literal[
        "speech",
        "thought",
        "shout",
        "whisper",
        "narration",
    ] = "speech"


class PanelSpec(BaseModel):
    panel_id: str = Field(min_length=1)
    bbox_norm: BBox = Field(
        description="(x, y, w, h) normalized to [0..1], origin top-left",
    )
    camera_angle: str = Field(min_length=1)
    shot: Literal[
        "establishing",
        "wide",
        "medium",
        "closeup",
        "extreme_closeup",
        "over_shoulder",
        "dynamic",
    ]
    scene: str = Field(min_length=1)
    visual_focus: str = Field(min_length=1)
    characters: List[str] = Field(default_factory=list)
    dialogue: List[DialogueLine] = Field(default_factory=list)
    sdxl_prompt: str = Field(min_length=1)
    sdxl_negative_prompt: str = Field(min_length=1)
    seed: Optional[int] = None
    steps: int = 50
    cfg_scale: float = 6.5


class PageSpec(BaseModel):
    page_index: int = Field(ge=0)
    page_width: int = 1536
    page_height: int = 2304
    gutter_px: int = 24
    border_px: int = 6
    panels: List[PanelSpec] = Field(min_length=1)


class Storyboard(BaseModel):
    version: str = "manga_ai.storyboard.v1"
    title: str = "Untitled"
    logline: str = Field(min_length=1)
    style_notes: str = Field(min_length=1)
    global_negative_prompt: str = Field(min_length=1)
    pages: List[PageSpec] = Field(min_length=1)


def storyboard_json_schema() -> Dict[str, Any]:
    return Storyboard.model_json_schema()


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return text[start : end + 1]


def _sanitize_common_json_issues(s: str) -> str:
    s = re.sub(r"\u0000", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _repair_json_like(text: str) -> str:
    s = _extract_json_object(text)
    s = re.sub(r"\u0000", "", s)
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r'"bbox_norm"\s*:\s*\(([^)]*)\)', r'"bbox_norm": [\1]', s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def _parse_json_or_python_object(decoded: str) -> Any:
    repaired = _repair_json_like(decoded)
    try:
        return json.loads(repaired)
    except Exception:
        pass

    py = repaired
    py = re.sub(r"\bnull\b", "None", py)
    py = re.sub(r"\btrue\b", "True", py)
    py = re.sub(r"\bfalse\b", "False", py)
    return ast.literal_eval(py)


def validate_storyboard(data: Any) -> Storyboard:
    try:
        sb = Storyboard.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Storyboard JSON failed validation: {e}") from e

    for page in sb.pages:
        for panel in page.panels:
            x, y, w, h = panel.bbox_norm
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                raise ValueError(f"Invalid bbox_norm for {panel.panel_id}: {panel.bbox_norm}")
            if x + w > 1.0 + 1e-6 or y + h > 1.0 + 1e-6:
                raise ValueError(f"bbox_norm out of bounds for {panel.panel_id}: {panel.bbox_norm}")

    return sb


def _apply_storyboard_defaults(data: Any) -> Any:
    if not isinstance(data, dict):
        return data

    def _clamp_bbox(b: Any) -> Any:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            return b
        try:
            x = float(b[0])
            y = float(b[1])
            w = float(b[2])
            h = float(b[3])
        except Exception:
            return b

        if not all(map(lambda v: (v == v) and abs(v) != float("inf"), [x, y, w, h])):
            return b

        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(1e-4, min(1.0, w))
        h = max(1e-4, min(1.0, h))

        if x + w > 1.0:
            w = max(1e-4, 1.0 - x)
        if y + h > 1.0:
            h = max(1e-4, 1.0 - y)

        if x + w > 1.0:
            x = max(0.0, 1.0 - w)
        if y + h > 1.0:
            y = max(0.0, 1.0 - h)

        return (x, y, w, h)

    global_neg = data.get("global_negative_prompt")
    if not isinstance(global_neg, str) or not global_neg.strip():
        global_neg = (
            "worst quality, low quality, lowres, blurry, jpeg artifacts, watermark, logo, signature, "
            "bad anatomy, bad hands, extra fingers, missing fingers, extra limbs, deformed"
        )
        data["global_negative_prompt"] = global_neg

    pages = data.get("pages")
    if not isinstance(pages, list):
        return data

    for page in pages:
        if not isinstance(page, dict):
            continue
        panels = page.get("panels")
        if not isinstance(panels, list):
            continue
        for panel in panels:
            if not isinstance(panel, dict):
                continue
            if "bbox_norm" in panel:
                panel["bbox_norm"] = _clamp_bbox(panel.get("bbox_norm"))
            neg = panel.get("sdxl_negative_prompt")
            if not isinstance(neg, str) or not neg.strip():
                panel["sdxl_negative_prompt"] = global_neg

    return data


def qwen_storyboard_system_prompt() -> str:
    schema = json.dumps(storyboard_json_schema(), ensure_ascii=False)
    return (
        "You are Qwen2.5 acting as a professional manga director and production planner. "
        "Given a short story prompt, you MUST output a single JSON object that matches the provided JSON Schema exactly. "
        "Do not output markdown. Do not include explanations. Output JSON only. "
        "First, you must fully develop the story beats for the page (scene context, character actions, and dialogue). "
        "Then, you must translate that narrative into a single manga page plan with strong cinematic paneling and clear staging. "
        "You MUST ask the image model (SDXL) to draw speech bubbles AND render the dialogue text inside the bubbles. "
        "The exact dialogue text must be short, legible, and match the dialogue lines you provide. "
        "Do not reserve blank bubble areas for later; the bubbles and text must be drawn directly in the image. "
        "Ensure each panel includes a clean SDXL prompt describing the artwork AND the speech bubbles with the exact dialogue text. "
        "Maintain consistent character design across all panels (same hair, outfit, accessories, age). "
        "Maintain stable background continuity (same location, consistent time of day) unless the script explicitly changes it. "
        "Use dynamic camera angles and varied shots while keeping the location readable. "
        "Use sharp manga lineart, anime shading, cinematic lighting, high detail. "
        "Avoid copyrighted characters, logos, watermarks, signatures. "
        "\n\nJSON Schema (for validation):\n"
        + schema
    )


def qwen_storyboard_user_prompt(story_prompt: str, pages: int = 1) -> str:
    return (
        f"Story prompt:\n{story_prompt.strip()}\n\n"
        f"Requirements:\n"
        f"- Produce exactly {pages} page(s).\n"
        "- Each page must have exactly 4 to 6 panels.\n"
        "- Provide normalized bboxes that form a clean page layout with gutters.\n"
        "- Dialogue must be short and readable.\n"
        "- The image must include speech bubbles and the dialogue text rendered inside them.\n"
        "- sdxl_prompt: include the scene + characters + composition AND include explicit instructions to draw speech bubbles containing the exact dialogue text.\n"
        "- Do NOT put dialogue only in the JSON dialogue field; also include it in sdxl_prompt so the image model draws it.\n"
        "- sdxl_negative_prompt: include watermark, logo, signature, lowres, blurry, bad anatomy, extra limbs.\n"
    )


def load_qwen(model_dir: Path, dtype: torch.dtype = torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, **base_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, **base_kwargs)
    model.eval()
    return tokenizer, model


def load_qwen_backend(model_dir: str | Path, dtype: torch.dtype = torch.float16) -> QwenBackend:
    model_dir = Path(model_dir)
    tokenizer, model = load_qwen(model_dir=model_dir, dtype=dtype)
    return QwenBackend(tokenizer=tokenizer, model=model)


def _build_chat_input(tokenizer, system: str, user: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(text, return_tensors="pt").input_ids

    joined = system + "\n\n" + user
    return tokenizer(joined, return_tensors="pt").input_ids


@torch.inference_mode()
def generate_storyboard(
    story_prompt: str,
    model_dir: str | Path,
    pages: int = 1,
    max_new_tokens: int = 1800,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> Storyboard:
    backend = load_qwen_backend(model_dir=model_dir)
    return generate_storyboard_from_backend(
        story_prompt=story_prompt,
        backend=backend,
        pages=pages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


@torch.inference_mode()
def generate_storyboard_from_backend(
    story_prompt: str,
    backend: QwenBackend,
    pages: int = 1,
    max_new_tokens: int = 1800,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> Storyboard:
    tokenizer = backend.tokenizer
    model = backend.model

    system = qwen_storyboard_system_prompt()
    base_user = qwen_storyboard_user_prompt(story_prompt=story_prompt, pages=pages)

    last_decoded: str = ""
    last_err: Optional[BaseException] = None
    for attempt in range(3):
        user = base_user
        do_sample = False

        if attempt > 0:
            user = (
                base_user
                + "\n\n"
                + "CRITICAL: Output a single JSON object ONLY. No markdown, no backticks, no explanation. "
                + "Your first character must be '{' and your last character must be '}'."
            )

        attempt_max_new_tokens = max_new_tokens
        if attempt == 1:
            attempt_max_new_tokens = int(max_new_tokens * 1.6)
        elif attempt >= 2:
            attempt_max_new_tokens = int(max_new_tokens * 2.2)

        input_ids = _build_chat_input(tokenizer, system=system, user=user).to(model.device)

        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": attempt_max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.05,
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
        }

        out = model.generate(**gen_kwargs)

        decoded = tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True)
        decoded = decoded.strip()
        last_decoded = decoded

        try:
            data = _parse_json_or_python_object(decoded)
            data = _apply_storyboard_defaults(data)
            return validate_storyboard(data)
        except Exception as e:
            last_err = e
            continue

    flat = last_decoded.replace("\n", " ").strip()
    head = flat[:300]
    tail = flat[-300:] if len(flat) > 600 else ""
    if tail:
        preview = head + " ... " + tail
    else:
        preview = head
    err_msg = repr(last_err) if last_err is not None else "Unknown parse error"
    raise ValueError(
        f"Failed to parse JSON from model output (after retries). Last error: {err_msg}. Output preview: {preview}"
    )


def save_storyboard(sb: Storyboard, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sb.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
