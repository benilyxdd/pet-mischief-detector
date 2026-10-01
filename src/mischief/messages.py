"""Resolve warning messages for a (pet, object, risk_level) triple.

Lookup order (first hit wins):
    1. messages.overrides[risk_level]["{pet}-{object}"]
    2. messages.templates[risk_level].format(pet=..., object=...)
    3. generic fallback string
"""
from __future__ import annotations


def format_message(
    messages_cfg: dict,
    risk_level: str,
    pet_label: str,
    object_label: str,
) -> str:
    overrides = (messages_cfg.get("overrides") or {}).get(risk_level) or {}
    key = f"{pet_label}-{object_label}"
    if key in overrides:
        return overrides[key]

    templates = messages_cfg.get("templates") or {}
    template = templates.get(risk_level)
    if template:
        return template.format(pet=pet_label, object=object_label)

    return f"{pet_label} and {object_label}: {risk_level} risk."
