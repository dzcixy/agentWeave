from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from agentweaver.utils.io import ensure_dir


def chakra_proto_available() -> bool:
    candidates = ["chakra", "et_def_pb2", "chakra.et_def_pb2"]
    for name in candidates:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except ModuleNotFoundError:
            continue
    return False


def export_proto_if_available(payload: dict[str, Any], out_dir: str | Path, prefix: str = "agentweaver") -> dict[str, Any]:
    out = Path(out_dir)
    ensure_dir(out)
    if not chakra_proto_available():
        manifest = {
            "CHAKRA_PROTO_EXPORT": "NOT_AVAILABLE",
            "reason": "Chakra protobuf Python package was not importable; intermediate JSON remains the authoritative export.",
            "json_node_count": len(payload.get("nodes", [])),
        }
        (out / f"{prefix}_proto_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**manifest, "proto_file_count": 0, "proto_files": []}
    # The repository does not vendor a stable Chakra proto schema wrapper. Emit a
    # schema-neutral manifest instead of inventing incompatible binary records.
    manifest = {
        "CHAKRA_PROTO_EXPORT": "NOT_IMPLEMENTED_SCHEMA_BINDING",
        "reason": "A Chakra protobuf package is importable, but no repository-local schema binding is configured.",
        "json_node_count": len(payload.get("nodes", [])),
    }
    (out / f"{prefix}_proto_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "proto_file_count": 0, "proto_files": []}
