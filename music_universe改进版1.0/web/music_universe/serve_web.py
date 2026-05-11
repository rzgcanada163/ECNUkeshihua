#!/usr/bin/env python3
"""在本目录启动静态文件服务，避免 file:// 与跨域问题（见《问题排查》3.1）。

可在任意 cwd 执行：
  python web/music_universe/serve_web.py
或先 cd 到本文件所在目录：
  python serve_web.py

环境变量 PORT 可改端口（默认 8080）。
"""

from __future__ import annotations

import http.server
import os
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8080"))


def main() -> None:
    os.chdir(ROOT)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving directory: {ROOT}")
        print(f"Open in browser: http://127.0.0.1:{PORT}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
