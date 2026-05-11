"""从 music_universe_data_embedded.js 导出 JSON，便于校验或「仅 fetch JSON」的部署（见《问题排查》3.3）。

用法（在仓库根目录 music_universe改进版1.0/ 下）::

    python code/export_site_data_json.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
JS_PATH = PROJECT_ROOT / "web" / "music_universe" / "music_universe_data_embedded.js"
JSON_PATH = PROJECT_ROOT / "web" / "music_universe" / "music_universe_data_embedded.json"
PREFIX = "window.SITE_DATA = "


def main() -> int:
    if not JS_PATH.is_file():
        print(f"找不到数据文件: {JS_PATH}", file=sys.stderr)
        return 1
    text = JS_PATH.read_text(encoding="utf-8-sig").lstrip()
    if not text.startswith(PREFIX):
        print(f"文件必须以 {PREFIX!r} 开头。", file=sys.stderr)
        return 1
    payload = text[len(PREFIX) :].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        print(f"JSON 解析失败: {exc}", file=sys.stderr)
        return 1
    JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {JSON_PATH}（{len(JSON_PATH.read_text(encoding='utf-8'))} 字符）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
