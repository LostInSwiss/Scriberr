"""Pinned file/config/map contract; pure checks do not import a recognition engine."""
from pathlib import Path
from collections.abc import Mapping, Sequence
from enum import Enum
import hashlib
import json
import os


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def json_value(value):
    """Build a JSON-native copy, preserving enum values without mutating configs."""
    # Enum must precede primitive checks: IntEnum/str-mixin Enum are subclasses.
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            key = json_value(key)
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError('Unsupported JSON mapping key: '+type(key).__name__)
            # Canonicalise exactly as JSON does and detect Enum/primitive collisions.
            encoded = key if isinstance(key, str) else json.dumps(key, allow_nan=False)
            if encoded in result:
                raise ValueError('JSON mapping key collision: '+encoded)
            result[encoded] = json_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if callable(getattr(value, 'item', None)):
        scalar = value.item()
        if scalar is not value:
            return json_value(scalar)
    raise TypeError('Unsupported JSON value: '+type(value).__name__)

def json_dumps(data, *, raw=False, indent=None):
    return json.dumps(json_value(data), ensure_ascii=False, allow_nan=raw, indent=indent)

def save(path, data, *, raw=False):
    # Validate/encode first so failure leaves no empty or partial evidence file.
    encoded = json_dumps(data, raw=raw, indent=2)
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(encoded+'\n')


def selected_plan(root):
    value=os.environ.get('SCRIBERR_SA1_PLAN_DIR')
    if not value:return Path(root)/PLAN_REL
    plan=Path(value).resolve()
    allowed=Path(root)/'evaluations/benchmarks/2026-11-17-scriberr-d1-vibe-three-v1/plans'
    if not plan.is_relative_to(allowed) or plan.parent!=allowed:
        raise ValueError('Unpinned per-clip plan route')
    return plan

