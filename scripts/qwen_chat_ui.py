from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.model_downloader import ensure_models_downloaded
from scripts.storyboard import (
    Storyboard,
    generate_storyboard_from_backend,
    load_qwen_backend,
    validate_storyboard,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_qwen_dir() -> Path:
    root = _project_root()
    return Path(os.environ.get("MANGA_AI_QWEN_DIR", str(root / "models" / "qwen2.5")))


def _build_chat_input(tokenizer: Any, messages: List[dict]) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt").input_ids

    joined = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    return tokenizer(joined, return_tensors="pt").input_ids


@torch.inference_mode()
def _chat_generate(
    backend,
    history: List[Tuple[str, str]],
    user_message: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[Tuple[str, str]], str]:
    tokenizer = backend.tokenizer
    model = backend.model

    system = (
        "You are Qwen, a helpful assistant for manga story development. "
        "When asked, you can expand prompts, propose story beats, character arcs, and scene direction."
    )

    messages: List[dict] = [{"role": "system", "content": system}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_message.strip()})

    input_ids = _build_chat_input(tokenizer, messages).to(model.device)

    do_sample = bool(temperature and temperature > 0)

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "temperature": float(temperature) if do_sample else None,
        "top_p": float(top_p) if do_sample else None,
        "repetition_penalty": 1.05,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
    }

    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    out = model.generate(**gen_kwargs)
    decoded = tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()

    new_history = history + [(user_message, decoded)]
    return new_history, ""


def _storyboard_summary(sb: Storyboard) -> str:
    lines: List[str] = []
    lines.append(f"Title: {sb.title}")
    lines.append(f"Logline: {sb.logline}")
    lines.append(f"Style notes: {sb.style_notes}")
    lines.append("")

    for page in sb.pages:
        lines.append(f"Page {page.page_index}: {len(page.panels)} panels")
        for panel in page.panels:
            lines.append(f"- Panel {panel.panel_id}: {panel.shot}, {panel.camera_angle}")
            if panel.dialogue:
                for d in panel.dialogue:
                    lines.append(f"  - [{d.bubble_style}] {d.speaker}: {d.text}")

    return "\n".join(lines)


def _generate_storyboard(backend, story_prompt: str, pages: int, max_new_tokens: int, temperature: float, top_p: float):
    sb = generate_storyboard_from_backend(
        story_prompt=story_prompt,
        backend=backend,
        pages=int(pages),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    sb_json = sb.model_dump_json(indent=2, ensure_ascii=False)
    summary = _storyboard_summary(sb)
    return sb_json, summary


def _load_backend_and_models() -> Any:
    qwen_dir = _default_qwen_dir()
    qwen_dir.mkdir(parents=True, exist_ok=True)

    ensure_models_downloaded(
        qwen_dir=qwen_dir,
        sdxl_dir=_project_root() / "models" / "sdxl",
        hf_token=os.environ.get("MANGA_AI_HF_TOKEN"),
    )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    backend = load_qwen_backend(model_dir=qwen_dir, dtype=dtype)
    return backend


def main() -> None:
    backend = _load_backend_and_models()

    with gr.Blocks(title="Qwen Manga Chat") as demo:
        gr.Markdown("# Qwen Manga Chat\nLocal Qwen chat + storyboard generator")

        backend_state = gr.State(backend)

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=520)
            msg = gr.Textbox(label="Message", lines=3)

            with gr.Row():
                send = gr.Button("Send")
                clear = gr.Button("Clear")

            max_new_tokens = gr.Slider(128, 2048, value=512, step=32, label="Max new tokens")
            temperature = gr.Slider(0.0, 1.2, value=0.4, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

            def _send(backend, history, message, max_new_tokens, temperature, top_p):
                history = history or []
                if message is None or not str(message).strip():
                    return history, ""
                return _chat_generate(
                    backend=backend,
                    history=history,
                    user_message=str(message),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

            send.click(_send, inputs=[backend_state, chatbot, msg, max_new_tokens, temperature, top_p], outputs=[chatbot, msg])
            msg.submit(_send, inputs=[backend_state, chatbot, msg, max_new_tokens, temperature, top_p], outputs=[chatbot, msg])
            clear.click(lambda: [], outputs=[chatbot])

        with gr.Tab("Storyboard"):
            story_prompt = gr.Textbox(label="Basic prompt", lines=8)
            pages = gr.Slider(1, 5, value=1, step=1, label="Pages")

            with gr.Row():
                sb_max_new_tokens = gr.Slider(512, 4096, value=1800, step=64, label="Max new tokens")
                sb_temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
                sb_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

            gen = gr.Button("Generate storyboard")
            sb_json = gr.Textbox(label="Storyboard JSON", lines=20)
            sb_summary = gr.Textbox(label="Summary", lines=14)

            gen.click(
                _generate_storyboard,
                inputs=[backend_state, story_prompt, pages, sb_max_new_tokens, sb_temperature, sb_top_p],
                outputs=[sb_json, sb_summary],
            )

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
