from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(v) for v in value]
    return value


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise SystemExit("YAML configs require PyYAML. Install pyyaml or use JSON.") from exc
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a mapping: {config_path}")
    return data


def merge_config(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def write_config(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plain = _to_plain_data(data)
    try:
        import yaml
    except Exception:
        path.write_text(json.dumps(plain, indent=2) + "\n", encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(plain, sort_keys=False), encoding="utf-8")
