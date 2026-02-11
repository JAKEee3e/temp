from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None:
        return default
    return v


def _file_looks_downloaded(path: Path, min_bytes: int) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _dir_looks_like_diffusers_model(model_dir: Path) -> bool:
    return (model_dir / "model_index.json").exists()


def _dir_looks_like_hf_transformers_model(model_dir: Path) -> bool:
    return (model_dir / "config.json").exists() and (
        (model_dir / "tokenizer.json").exists()
        or (model_dir / "tokenizer.model").exists()
        or (model_dir / "tokenizer_config.json").exists()
    )


def _download_file(url: str, out_path: Path) -> None:
    try:
        import requests
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency 'requests'. Install it (pip install requests) or use the HuggingFace snapshot downloader path."
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)

    tmp.replace(out_path)


def _snapshot_download_repo(repo_id: str, local_dir: Path, token: Optional[str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install requirements (pip install -r requirements.txt)."
        ) from e

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to download model from HuggingFace. "
            "If this is a gated model, you must (1) accept the license on HuggingFace and (2) set MANGA_AI_HF_TOKEN. "
            f"Repo: {repo_id}"
        ) from e


def ensure_models_downloaded(
    *,
    qwen_dir: Path,
    sdxl_dir: Path,
    hf_token: Optional[str] = None,
) -> None:
    hf_token = hf_token if hf_token is not None else os.environ.get("MANGA_AI_HF_TOKEN")

    qwen_repo = _env("MANGA_AI_QWEN_REPO", "Qwen/Qwen2.5-7B-Instruct")

    if not _dir_looks_like_hf_transformers_model(qwen_dir):
        _snapshot_download_repo(repo_id=qwen_repo, local_dir=qwen_dir, token=hf_token)

    # SDXL / Animagine handling removed; this function now only ensures the Qwen model is downloaded.
