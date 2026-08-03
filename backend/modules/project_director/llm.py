import json
import os
from typing import List, Optional

import httpx

from ..prompt_builder.controls import AssetType, View, Animation, Palette, SpriteSize
from .director import ProjectDirector, ProjectPlan, PlanStep, _generate_title

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = """You are an AI Project Director for a game asset pipeline. You convert a natural-language request into a structured JSON plan for generating sprite assets.

Return ONLY a JSON object with this shape:
{
  "title": "short title for the project",
  "steps": [
    {
      "asset_type": "character",
      "view": "front",
      "animation": "idle",
      "palette": "auto",
      "sprite_size": "32x32",
      "theme": "forest knight",
      "num_frames": 4,
      "seed": -1
    }
  ]
}

Rules:
- asset_type is one of: character, building, weapon, vehicle, tree, road, ui, icon, enemy, prop, tileset, projectile, effect
- view is one of: front, side, top, isometric, 3/4, back
- animation is one of: idle, walk, run, attack, hurt, death, jump, shoot, cast, none
- palette is one of: auto, retro_8, retro_16, retro_32, monochrome, gameboy, snes, custom
- sprite_size is one of: 16x16, 32x32, 64x64, 128x128
- num_frames is a positive integer
- seed is -1 for random
- theme is a short phrase describing the visual theme
- Split multi-asset requests into one step per asset.
- Respond with valid JSON only, no markdown fences."""


def _coerce_enum(enum_cls, value, default):
    if value is None:
        return default
    s = str(value).strip().lower()
    for member in enum_cls:
        if member.value.lower() == s or member.name.lower() == s:
            return member
    return default


def _coerce_int(value, default, min_value=None):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None and v < min_value:
        return min_value
    return v


def _plan_from_llm_dict(raw: dict, request: str) -> ProjectPlan:
    steps_raw = raw.get("steps", [])
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError("LLM returned no steps")

    steps: List[PlanStep] = []
    for item in steps_raw:
        if not isinstance(item, dict):
            continue
        steps.append(
            PlanStep(
                asset_type=_coerce_enum(AssetType, item.get("asset_type"), AssetType.CHARACTER),
                view=_coerce_enum(View, item.get("view"), View.FRONT),
                animation=_coerce_enum(Animation, item.get("animation"), Animation.IDLE),
                palette=_coerce_enum(Palette, item.get("palette"), Palette.AUTO),
                sprite_size=_coerce_enum(SpriteSize, item.get("sprite_size"), SpriteSize.S_32),
                theme=str(item.get("theme") or ""),
                num_frames=_coerce_int(item.get("num_frames"), 1, min_value=1),
                seed=_coerce_int(item.get("seed"), -1),
            )
        )

    if not steps:
        raise ValueError("LLM returned no valid steps")

    title = str(raw.get("title") or "").strip()
    if not title:
        title = _generate_title(request, steps)
    return ProjectPlan(title=title, steps=steps)


class LLMProjectDirector(ProjectDirector):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        timeout: float = 30.0,
    ):
        super().__init__()
        self.api_key = (
            api_key
            or os.environ.get("PROJECT_DIRECTOR_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = base_url or os.environ.get("PROJECT_DIRECTOR_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.environ.get("PROJECT_DIRECTOR_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self._http_client = http_client

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        return httpx.Client(timeout=self.timeout)

    def parse(self, request: str) -> ProjectPlan:
        if not self.llm_enabled:
            return super().parse(request)
        try:
            plan = self._parse_with_llm(request)
            if plan is not None and plan.steps:
                return plan
        except Exception:
            pass
        return super().parse(request)

    def _parse_with_llm(self, request: str) -> Optional[ProjectPlan]:
        owned_client = self._http_client is None
        client = self._get_client()
        try:
            response = client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": request},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            raw = json.loads(content)
            return _plan_from_llm_dict(raw, request)
        finally:
            if owned_client:
                client.close()
