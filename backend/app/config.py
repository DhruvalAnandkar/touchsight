"""Application settings loaded from the environment and an optional .env file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from pipeline.page import PAGE_MARGIN_MM, PAGE_MM, TactileConfig

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_ROOT.parent


def _env_files() -> tuple[str, ...]:
    candidates = (
        Path.cwd() / ".env",
        _BACKEND_ROOT / ".env",
        _REPO_ROOT / ".env",
    )
    return tuple(str(path) for path in candidates if path.is_file())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files() or None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = _BACKEND_ROOT / "data"
    demo_cache_dir: Path = _BACKEND_ROOT / "demo_cache"
    max_upload_bytes: int = 8_000_000
    max_regions: int = Field(default=8, ge=1, le=8)
    page_size: str = "letter"
    dpi: int = Field(default=120, ge=72, le=300)
    margin_mm: float = PAGE_MARGIN_MM
    min_stroke_mm: float = 1.5
    min_gap_mm: float = 2.5
    min_area_mm2: float = 40.0
    epsilon_mm: float = 0.55
    sam_api_token: str = ""
    sam_model: str = "meta/sam-2"
    sam_api_base: str = "https://api.replicate.com/v1"
    sam_timeout_s: float = 30.0
    vlm_provider: str = ""
    vlm_model: str = ""
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_answer_model: str = "gpt-4o-mini"
    cors_origins: str = "http://localhost:5173"

    def vlm(self):
        from pipeline.understand import VlmConfig

        if os.environ.get("PYTEST_CURRENT_TEST"):
            return VlmConfig()
        provider = self.vlm_provider.strip().lower()
        if provider == "anthropic":
            key = self.vlm_api_key.strip() or self.anthropic_api_key.strip()
        else:
            key = self.vlm_api_key.strip() or self.openai_api_key.strip()
        if not provider:
            key = ""
        base = self.vlm_base_url.strip() or self.openai_base_url
        return VlmConfig(
            provider=provider,
            model=self.vlm_model,
            api_key=key,
            base_url=base,
            cache_dir=self.data_dir / "vlm",
        )

    def diagram_model(self) -> tuple[str, str, str]:
        """Key, base URL, and model for spoken diagram questions.

        Tests never use the vision key from the environment. A test that wants
        the HTTP path sets openai_api_key on the Settings object.
        """

        if os.environ.get("PYTEST_CURRENT_TEST"):
            key = self.openai_api_key.strip()
            if not key:
                return "", "", ""
            base = self.openai_base_url.strip() or "https://api.openai.com/v1"
            model = self.openai_answer_model.strip() or "gpt-4o-mini"
            return key, base, model

        key = self.openai_api_key.strip()
        if key:
            base = self.openai_base_url.strip() or "https://api.openai.com/v1"
            model = self.openai_answer_model.strip() or "gpt-4o-mini"
            return key, base, model

        vision_key = self.vlm_api_key.strip()
        vision_base = self.vlm_base_url.strip()
        vision_model = next((item.strip() for item in self.vlm_model.split(",") if item.strip()), "")
        if vision_key and vision_base and vision_model:
            return vision_key, vision_base, vision_model
        return "", "", ""

    def tactile(self) -> TactileConfig:
        page = self.page_size.lower()
        if page not in PAGE_MM:
            page = "letter"
        return TactileConfig(
            page=page,  # type: ignore[arg-type]
            dpi=self.dpi,
            margin_mm=self.margin_mm,
            min_stroke_mm=self.min_stroke_mm,
            min_gap_mm=self.min_gap_mm,
            min_area_mm2=self.min_area_mm2,
            epsilon_mm=self.epsilon_mm,
        )

    def origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
