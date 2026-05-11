"""Enrich Music Universe web data with public music metadata.

This script reads web/music_universe/music_universe_data_embedded.js, queries the
iTunes Search API for artwork and legal 30-second preview URLs, then writes the
enriched data back to the web file. It also stores a CSV/JSON cache under output
so repeated runs are fast and resilient.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WEB_DATA_PATH = PROJECT_ROOT / "web" / "music_universe" / "music_universe_data_embedded.js"
CACHE_JSON_PATH = PROJECT_ROOT / "output" / "tables" / "itunes_metadata_cache.json"
CACHE_CSV_PATH = PROJECT_ROOT / "output" / "tables" / "itunes_metadata_enriched.csv"
JS_PREFIX = "window.SITE_DATA = "
# 去除易干扰 iTunes/Deezer 查询的控制字符与零宽字符（见《问题排查》2.3）。
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u202a-\u202e\ufeff]")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text.replace(";", " ")).strip()


def normalized_key(item: dict[str, Any]) -> str:
    title = clean_text(item.get("title")).lower()
    artist = clean_text(item.get("artist")).lower()
    return f"{title}::{artist}"


def make_query(item: dict[str, Any]) -> str:
    parts = [
        clean_text(item.get("title")),
        clean_text(item.get("artist")),
    ]
    album = clean_text(item.get("album"))
    if album and not album.lower().startswith("billboard #1"):
        parts.append(album)
    return " ".join(part for part in parts if part)


def load_site_data(path: Path) -> dict[str, Any]:
    # utf-8-sig：兼容带 BOM 的编辑器保存；lstrip 容忍文件开头空白。
    text = path.read_text(encoding="utf-8-sig").lstrip()
    if not text.startswith(JS_PREFIX):
        raise ValueError(
            f"{path} does not start with {JS_PREFIX!r} (after BOM/whitespace). "
            "请以 window.SITE_DATA = 开头保存 JSON 内容。"
        )
    payload = text[len(JS_PREFIX) :].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    return json.loads(payload)


def save_site_data(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        JS_PREFIX + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def save_cache(cache: dict[str, dict[str, Any]]) -> None:
    CACHE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_JSON_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for key, value in sorted(cache.items()):
        row = {"cache_key": key}
        row.update(value)
        rows.append(row)
    fieldnames = [
        "cache_key",
        "found",
        "query",
        "track_name",
        "artist_name",
        "collection_name",
        "preview_url",
        "artwork_url",
        "track_view_url",
        "source",
    ]
    with CACHE_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def request_itunes(query: str, country: str = "US") -> dict[str, Any]:
    params = urlencode(
        {
            "media": "music",
            "entity": "song",
            "limit": 5,
            "country": country,
            "term": query,
        }
    )
    url = f"https://itunes.apple.com/search?{params}"
    request = Request(url, headers={"User-Agent": "MusicUniverseCourseProject/1.0"})
    with urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def request_deezer(query: str) -> dict[str, Any]:
    params = urlencode({"q": query, "limit": 5})
    url = f"https://api.deezer.com/search?{params}"
    request = Request(url, headers={"User-Agent": "MusicUniverseCourseProject/1.0"})
    with urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def score_result(item: dict[str, Any], result: dict[str, Any]) -> int:
    title = clean_text(item.get("title")).lower()
    artist = clean_text(item.get("artist")).lower()
    album = clean_text(item.get("album")).lower()
    result_title = clean_text(result.get("trackName")).lower()
    result_artist = clean_text(result.get("artistName")).lower()
    result_album = clean_text(result.get("collectionName")).lower()

    score = 0
    if title and title == result_title:
        score += 80
    elif title and (title in result_title or result_title in title):
        score += 42
    if artist and result_artist and (artist in result_artist or result_artist in artist):
        score += 44
    if album and result_album and (album in result_album or result_album in album):
        score += 18
    if result.get("previewUrl"):
        score += 10
    if result.get("artworkUrl100"):
        score += 6
    return score


def lookup_metadata(item: dict[str, Any], country: str) -> dict[str, Any]:
    query = make_query(item)
    if not query:
        return {"found": False, "query": query, "source": "itunes"}
    try:
        data = request_itunes(query, country=country)
    except Exception as exc:
        return {"found": False, "query": query, "source": "itunes", "error": str(exc)}

    results = data.get("results") or []
    if not results:
        fallback_query = " ".join([clean_text(item.get("title")), clean_text(item.get("artist"))]).strip()
        if fallback_query and fallback_query != query:
            try:
                data = request_itunes(fallback_query, country=country)
                results = data.get("results") or []
                query = fallback_query
            except Exception:
                results = []

    if not results:
        return {"found": False, "query": query, "source": "itunes"}

    best = max(results, key=lambda result: score_result(item, result))
    artwork = clean_text(best.get("artworkUrl100"))
    if artwork:
        artwork = artwork.replace("100x100bb", "600x600bb")

    return {
        "found": True,
        "query": query,
        "track_name": clean_text(best.get("trackName")),
        "artist_name": clean_text(best.get("artistName")),
        "collection_name": clean_text(best.get("collectionName")),
        "preview_url": clean_text(best.get("previewUrl")),
        "artwork_url": artwork,
        "track_view_url": clean_text(best.get("trackViewUrl")),
        "source": "itunes",
    }


def lookup_deezer_metadata(item: dict[str, Any]) -> dict[str, Any]:
    query = " ".join([clean_text(item.get("title")), clean_text(item.get("artist"))]).strip()
    if not query:
        return {"found": False, "query": query, "source": "deezer"}
    try:
        data = request_deezer(query)
    except Exception as exc:
        return {"found": False, "query": query, "source": "deezer", "error": str(exc)}

    results = data.get("data") or []
    if not results:
        return {"found": False, "query": query, "source": "deezer"}

    def deezer_score(result: dict[str, Any]) -> int:
        wrapped = {
            "trackName": result.get("title"),
            "artistName": (result.get("artist") or {}).get("name"),
            "collectionName": (result.get("album") or {}).get("title"),
            "previewUrl": result.get("preview"),
            "artworkUrl100": (result.get("album") or {}).get("cover_medium"),
        }
        return score_result(item, wrapped)

    best = max(results, key=deezer_score)
    album = best.get("album") or {}
    artist = best.get("artist") or {}
    artwork = clean_text(album.get("cover_xl") or album.get("cover_big") or album.get("cover_medium"))
    return {
        "found": bool(best.get("preview") or artwork),
        "query": query,
        "track_name": clean_text(best.get("title")),
        "artist_name": clean_text(artist.get("name")),
        "collection_name": clean_text(album.get("title")),
        "preview_url": clean_text(best.get("preview")),
        "artwork_url": artwork,
        "track_view_url": clean_text(best.get("link")),
        "source": "deezer",
    }


def lookup_public_metadata(item: dict[str, Any], country: str) -> dict[str, Any]:
    meta = lookup_metadata(item, country=country)
    if meta.get("preview_url") and meta.get("artwork_url"):
        return meta
    deezer_meta = lookup_deezer_metadata(item)
    if deezer_meta.get("preview_url") or (not meta.get("found") and deezer_meta.get("artwork_url")):
        return deezer_meta
    return meta


def apply_metadata(items: list[dict[str, Any]], cache: dict[str, dict[str, Any]]) -> int:
    updated = 0
    for item in items:
        key = normalized_key(item)
        meta = cache.get(key, {})
        if not meta.get("found"):
            continue
        if meta.get("preview_url"):
            item["preview_url"] = meta["preview_url"]
            updated += 1
        if meta.get("artwork_url"):
            item["image_url"] = meta["artwork_url"]
            item["artwork_url"] = meta["artwork_url"]
        if meta.get("track_view_url"):
            item["itunes_url"] = meta["track_view_url"]
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich Music Universe metadata via iTunes Search.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of uncached unique songs to query. 0 = all.")
    parser.add_argument("--country", default="US", help="iTunes storefront country code.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Delay between network requests (larger = gentler on public APIs).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent lookup workers (lower = fewer rate-limit errors).",
    )
    parser.add_argument("--retry-failed", action="store_true", help="Retry cached entries without preview URLs.")
    args = parser.parse_args()

    data = load_site_data(WEB_DATA_PATH)
    all_items = list(data.get("spotifyItems", [])) + list(data.get("billboardItems", []))
    cache = load_cache(CACHE_JSON_PATH)

    unique: dict[str, dict[str, Any]] = {}
    for item in all_items:
        key = normalized_key(item)
        if key and key not in unique:
            unique[key] = item

    uncached = []
    for key, item in unique.items():
        cached = cache.get(key)
        if cached is None or (args.retry_failed and not cached.get("preview_url")):
            uncached.append((key, item))
    if args.limit > 0:
        uncached = uncached[: args.limit]

    print(f"Unique songs: {len(unique)} | cached: {len(cache)} | querying: {len(uncached)}")
    if uncached:
        workers = max(1, min(args.workers, 16))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for key, item in uncached:
                futures[executor.submit(lookup_public_metadata, item, args.country)] = key
                if args.sleep > 0:
                    time.sleep(args.sleep)

            for index, future in enumerate(as_completed(futures), start=1):
                key = futures[future]
                try:
                    cache[key] = future.result()
                except Exception as exc:
                    cache[key] = {"found": False, "source": "itunes", "error": str(exc)}
                if index % 50 == 0 or index == len(uncached):
                    found = sum(1 for value in cache.values() if value.get("found"))
                    previews = sum(1 for value in cache.values() if value.get("preview_url"))
                    print(f"Queried {index}/{len(uncached)} | found={found} | previews={previews}")
                    save_cache(cache)

    preview_count = apply_metadata(data.get("spotifyItems", []), cache)
    preview_count += apply_metadata(data.get("billboardItems", []), cache)
    save_site_data(WEB_DATA_PATH, data)
    save_cache(cache)

    artwork_count = sum(1 for item in all_items if clean_text(item.get("image_url")))
    print(f"Done. Items with preview_url: {preview_count}; items with image_url/artwork: {artwork_count}.")
    print(f"Updated: {WEB_DATA_PATH}")
    print(f"Cache:   {CACHE_JSON_PATH}")


if __name__ == "__main__":
    main()
