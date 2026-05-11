import os
import re
import json
import math
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from PIL import Image
import requests
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 1. 用户可直接修改的路径配置
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
_DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
# 默认使用仓库内 data/，换机器无需改代码；仍可用环境变量覆盖。
SPOTIFY_PATH = os.environ.get("MUSIC_SPOTIFY_PATH", str(_DEFAULT_DATA_DIR / "spotify_tracks.csv"))
BILLBOARD_PATH = os.environ.get("MUSIC_BILLBOARD_PATH", str(_DEFAULT_DATA_DIR / "billboard.csv"))
TOPICS_PATH = os.environ.get("MUSIC_TOPICS_PATH", str(_DEFAULT_DATA_DIR / "topics.csv"))
OUTPUT_ROOT = os.environ.get("MUSIC_OUTPUT_ROOT", str(PROJECT_ROOT / "output"))

# 可选：专辑图片映射表（支持列：track_id, image_path, image_url）；不存在则跳过相关图。
ALBUM_ART_MAP_PATH = os.environ.get(
    "MUSIC_ALBUM_ART_MAP_PATH", str(_DEFAULT_DATA_DIR / "album_art_map.csv")
)

# Plotly 图表字体：含西文与中文常见回退，减轻交互 HTML 中文乱码（见《问题排查》4.2）。
PLOTLY_CHART_FONT = dict(
    family=(
        "system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',"
        "'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif"
    ),
    size=12,
)


# =========================================================
# 2. 全局样式设置：让图尽量更适合课程论文展示
# =========================================================
PALETTE_MAIN = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
]

PALETTE_SOFT = [
    "#6C8EBF", "#F4A259", "#E76F51", "#5FB3B3", "#6AB187",
    "#E9C46A", "#A68FBF", "#F7A8B8", "#B08968", "#C7C7C7"
]

# 主题固定配色：保证不同图之间颜色语义一致
TOPIC_COLOR_FIXED = {
    "Love": "#E63946",
    "Lost Love": "#457B9D",
    "Longing for Love": "#1D3557",
    "Bad Relationships": "#6D597A",
    "Infidelity": "#9D4EDD",
    "Lust/Sex": "#06D6A0",
    "Dancing": "#00B4D8",
    "Partying": "#FF9F1C",
    "Bragging": "#F77F00",
    "Empowerment": "#2A9D8F",
    "Badassery": "#4361EE",
    "Death": "#6C757D",
    "Murder": "#495057",
    "Dreaming": "#80ED99",
    "Flying": "#4CC9F0",
    "Christmas": "#2D6A4F",
    "Nostalgia": "#B56576",
    "Heartbreak": "#8D99AE",
    "Loneliness": "#7B2CBF",
}

TOPIC_COLOR_FALLBACK = [
    "#FF6B6B", "#4ECDC4", "#FFD166", "#06D6A0", "#118AB2", "#EF476F",
    "#8338EC", "#3A86FF", "#FB8500", "#2A9D8F", "#E76F51", "#90BE6D",
    "#F28482", "#84A59D", "#8E9AAF", "#C77DFF", "#72EFDD", "#F4A261"
]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """将十六进制颜色转换为 rgba 字符串。"""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return f"rgba(107,114,128,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_topic_color_map(topics: List[str]) -> dict:
    """根据主题名称生成稳定颜色映射。"""
    cleaned = []
    for t in topics:
        if pd.isna(t):
            continue
        s = str(t).strip()
        if s:
            cleaned.append(s)
    unique_topics = sorted(set(cleaned))

    color_map = {}
    for t in unique_topics:
        if t in TOPIC_COLOR_FIXED:
            color_map[t] = TOPIC_COLOR_FIXED[t]

    unknown = [t for t in unique_topics if t not in color_map]
    for i, t in enumerate(unknown):
        color_map[t] = TOPIC_COLOR_FALLBACK[i % len(TOPIC_COLOR_FALLBACK)]
    return color_map


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "savefig.dpi": 300,
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


# =========================================================
# 3. 文件夹与日志
# =========================================================
def build_output_dirs(output_root: str) -> dict:
    """创建输出目录树，并返回常用目录字典。"""
    root = Path(output_root)
    dirs = {
        "root": root,
        "cleaned": root / "cleaned_data",
        "tables": root / "tables",
        "fig_spotify": root / "figures" / "spotify",
        "fig_billboard": root / "figures" / "billboard",
        "fig_comparison": root / "figures" / "comparison",
        "fig_html": root / "figures" / "interactive_html",
        "reports": root / "reports",
        "logs": root / "logs",
    }
    for p in dirs.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_path: Path) -> None:
    """设置日志，便于排查运行问题。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )


# =========================================================
# 4. 数据读取与清洗
# =========================================================
def safe_read_csv(path: str) -> pd.DataFrame:
    """读取 CSV：多编码回退（见《问题排查》1.2）。"""
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "cp932", "big5", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            logging.info(f"读取成功: {path} | encoding={enc} | shape={df.shape}")
            return df
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"无法读取文件: {path}\n最后错误: {last_err}")


def normalize_bool_text(series: pd.Series) -> pd.Series:
    """将 TRUE/FALSE/0/1 等形式尽量统一为 0/1。"""
    s = series.astype(str).str.strip().str.lower()
    mapper = {
        "true": 1, "false": 0,
        "1": 1, "0": 0,
        "yes": 1, "no": 0,
        "nan": np.nan, "none": np.nan, "na": np.nan
    }
    return s.map(mapper).astype("float")


def standardize_text_col(series: pd.Series) -> pd.Series:
    """统一文本空格和缺失。"""
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "NA": np.nan, "None": np.nan})
    )


def clean_spotify(df: pd.DataFrame) -> pd.DataFrame:
    """Spotify 数据清洗。"""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    expected_numeric = [
        "popularity", "duration_ms", "danceability", "energy", "key", "loudness",
        "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
        "valence", "tempo", "time_signature"
    ]
    for col in expected_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "explicit" in df.columns:
        df["explicit_num"] = normalize_bool_text(df["explicit"]).fillna(0)

    for col in ["track_id", "artists", "album_name", "track_name", "track_genre"]:
        if col in df.columns:
            df[col] = standardize_text_col(df[col])

    # 去重：优先按 track_id；若不存在则按歌曲名+艺人+专辑。
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"])
    else:
        keep_cols = [c for c in ["track_name", "artists", "album_name"] if c in df.columns]
        if keep_cols:
            df = df.drop_duplicates(subset=keep_cols)

    # 只保留关键特征非空的数据。
    core_cols = [c for c in ["popularity", "danceability", "energy", "valence", "tempo", "track_genre"] if c in df.columns]
    df = df.dropna(subset=core_cols)

    # 基于分位数构造热度层级，适合画分组对比图。
    if "popularity" in df.columns:
        df["popularity_tier"] = pd.qcut(
            df["popularity"].rank(method="first"),
            q=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["Low", "Mid-Low", "Mid-High", "High"]
        )

        high_cut = df["popularity"].quantile(0.90)
        low_cut = df["popularity"].quantile(0.10)
        df["hit_group"] = np.select(
            [df["popularity"] >= high_cut, df["popularity"] <= low_cut],
            ["Hit", "Low-Impact"],
            default="Middle"
        )

    if "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"] / 60000.0

    return df.reset_index(drop=True)


def clean_billboard(df: pd.DataFrame) -> pd.DataFrame:
    """Billboard 冠军歌曲数据清洗。"""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    text_cols = [
        "song", "artist", "label", "parent_label", "cdr_genre", "cdr_style",
        "discogs_genre", "discogs_style", "lyrical_topic", "lyrics"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = standardize_text_col(df[col])

    num_cols = [
        "weeks_at_number_one", "overall_rating", "divisiveness", "front_person_age",
        "bpm", "energy", "danceability", "happiness", "loudness_d_b",
        "acousticness", "length_sec"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["year"] = df["date"].dt.year
        df["decade"] = (df["year"] // 10) * 10

    # 显式字段标准化
    if "explicit" in df.columns:
        df["explicit_num"] = normalize_bool_text(df["explicit"]) 

    # 周数分组
    if "weeks_at_number_one" in df.columns:
        df["weeks_group"] = pd.cut(
            df["weeks_at_number_one"],
            bins=[0, 1, 3, 6, 12, 100],
            labels=["1 week", "2-3 weeks", "4-6 weeks", "7-12 weeks", "13+ weeks"],
            include_lowest=True
        )

    return df.reset_index(drop=True)


def clean_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Topics 数据清洗。若字段名有差异，也尽量保留。"""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = standardize_text_col(df[c])
    return df.reset_index(drop=True)


# =========================================================
# 5. 通用工具函数
# =========================================================
def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """保存数据表。"""
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info(f"已保存表格: {path}")


def save_figure(path: Path, tight: bool = True) -> None:
    """统一保存图片。"""
    if tight:
        plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logging.info(f"已保存图像: {path}")


def run_plot(plot_name: str, plot_fn) -> None:
    """执行绘图任务并在失败时继续流程。"""
    try:
        plot_fn()
    except Exception:
        logging.exception(f"绘图失败: {plot_name}")


def top_n_with_other(df: pd.DataFrame, cat_col: str, n: int = 10, value_name: str = "count") -> pd.DataFrame:
    """将长尾类别合并到 Other，便于图更干净。"""
    vc = df[cat_col].value_counts(dropna=False)
    top_index = vc.head(n).index
    temp = df.copy()
    temp[cat_col] = np.where(temp[cat_col].isin(top_index), temp[cat_col], "Other")
    out = temp[cat_col].value_counts().rename_axis(cat_col).reset_index(name=value_name)
    return out


def get_feature_columns_spotify(df: pd.DataFrame) -> List[str]:
    candidates = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo"
    ]
    return [c for c in candidates if c in df.columns]


def get_feature_columns_billboard(df: pd.DataFrame) -> List[str]:
    candidates = [
        "energy", "danceability", "happiness", "loudness_d_b", "acousticness", "bpm"
    ]
    return [c for c in candidates if c in df.columns]


def ensure_pca_projection(df: pd.DataFrame, feature_cols: List[str], prefix: str = "pca") -> pd.DataFrame:
    """对指定特征做 PCA，输出 2 维坐标。"""
    out = df.copy()
    if len(feature_cols) < 2:
        out[f"{prefix}_1"] = np.nan
        out[f"{prefix}_2"] = np.nan
        return out

    temp = out[feature_cols].dropna().copy()
    if temp.empty:
        out[f"{prefix}_1"] = np.nan
        out[f"{prefix}_2"] = np.nan
        return out

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(temp)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)
    proj = pd.DataFrame(coords, index=temp.index, columns=[f"{prefix}_1", f"{prefix}_2"])
    out = out.join(proj)
    return out


def sanitize_filename(name: str) -> str:
    """处理文件名非法字符。"""
    name = re.sub(r'[\\/:*?"<>|]+', '_', str(name))
    return name.strip()[:150]


# =========================================================
# 6. 专辑图相关：模板 / 读取 / 下载 / 显示
# =========================================================
def create_album_art_template(spotify_df: pd.DataFrame, template_path: Path, n: int = 200) -> None:
    """生成专辑图片映射模板，便于你后续手动补图。"""
    cols = [c for c in ["track_id", "track_name", "artists", "album_name", "popularity", "track_genre"] if c in spotify_df.columns]
    sample = spotify_df.sort_values("popularity", ascending=False)[cols].head(n).copy()
    sample["image_path"] = ""
    sample["image_url"] = ""
    save_dataframe(sample, template_path)


def read_album_art_map(path: str) -> Optional[pd.DataFrame]:
    """读取专辑图映射表，不存在则返回 None。"""
    if not os.path.exists(path):
        logging.warning("未检测到 album_art_map.csv，将跳过封面图相关可视化。")
        return None

    art = safe_read_csv(path)
    art.columns = [c.strip() for c in art.columns]
    required_any = {"image_path", "image_url"}
    if "track_id" not in art.columns or not required_any.intersection(set(art.columns)):
        logging.warning("album_art_map.csv 缺少必要列，至少需要 track_id + (image_path 或 image_url)。")
        return None

    for c in art.columns:
        if art[c].dtype == object:
            art[c] = standardize_text_col(art[c])
    return art


def download_image_if_needed(image_url: str, cache_dir: Path, file_stem: str) -> Optional[Path]:
    """若用户提供了 image_url，则尝试下载到本地缓存。"""
    if not image_url or pd.isna(image_url):
        return None

    safe_name = sanitize_filename(file_stem) + ".jpg"
    save_path = cache_dir / safe_name
    if save_path.exists():
        return save_path

    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        save_path.write_bytes(resp.content)
        return save_path
    except Exception as exc:
        logging.warning(f"图片下载失败: {image_url} | {exc}")
        return None


def resolve_album_art_path(row: pd.Series, cache_dir: Path) -> Optional[Path]:
    """优先使用 image_path；若无则尝试下载 image_url。"""
    image_path = row.get("image_path")
    image_url = row.get("image_url")

    if isinstance(image_path, str) and image_path.strip() and os.path.exists(image_path):
        return Path(image_path)

    if isinstance(image_url, str) and image_url.strip():
        stem = f"{row.get('track_id', 'unknown')}_{row.get('track_name', 'track')}"
        return download_image_if_needed(image_url, cache_dir, stem)

    return None


def merge_album_art(spotify_df: pd.DataFrame, art_df: Optional[pd.DataFrame], cache_dir: Path) -> pd.DataFrame:
    """将专辑图路径并入 Spotify 数据。"""
    out = spotify_df.copy()
    out["resolved_image_path"] = np.nan

    if art_df is None:
        return out

    merged = out.merge(art_df, on="track_id", how="left")
    resolved_paths = []
    for _, row in merged.iterrows():
        p = resolve_album_art_path(row, cache_dir)
        resolved_paths.append(str(p) if p else np.nan)
    merged["resolved_image_path"] = resolved_paths
    return merged


# =========================================================
# 7. Spotify 数据静态图
# =========================================================
def plot_spotify_popularity_distribution(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(12, 7))
    sns.histplot(df["popularity"], bins=30, kde=True, color=PALETTE_MAIN[0], edgecolor="white", alpha=0.85)
    plt.title("Spotify 热度分布")
    plt.xlabel("热度")
    plt.ylabel("数量")
    save_figure(out_path)


def plot_spotify_genre_counts(df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    temp = top_n_with_other(df, "track_genre", n=top_n, value_name="count")
    temp = temp.sort_values("count", ascending=True)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=temp, x="count", y="track_genre", color=PALETTE_MAIN[0])
    plt.title(f"歌曲数量前 {top_n} 的流派")
    plt.xlabel("歌曲数量")
    plt.ylabel("流派")
    save_figure(out_path)


def plot_spotify_genre_popularity(df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    genre_order = df["track_genre"].value_counts().head(top_n).index.tolist()
    temp = (
        df[df["track_genre"].isin(genre_order)]
        .groupby("track_genre", as_index=False)
        .agg(avg_popularity=("popularity", "mean"), count=("popularity", "size"))
        .sort_values("avg_popularity", ascending=False)
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(data=temp, x="avg_popularity", y="track_genre", color=PALETTE_SOFT[1])
    plt.title("各流派平均热度")
    plt.xlabel("平均热度")
    plt.ylabel("流派")
    save_figure(out_path)


def plot_spotify_boxplots_by_hit_group(df: pd.DataFrame, out_path: Path) -> None:
    feats = [c for c in ["danceability", "energy", "valence", "acousticness", "tempo"] if c in df.columns]
    if not feats:
        logging.warning("缺少可用于箱线图的特征列，跳过绘图。")
        return

    group_order = ["Hit", "Middle", "Low-Impact"]
    palette = {"Hit": "#E15759", "Middle": "#4E79A7", "Low-Impact": "#59A14F"}
    feat_label = {
        "danceability": "舞动性",
        "energy": "能量",
        "valence": "愉悦度",
        "acousticness": "原声占比",
        "tempo": "节奏速度"
    }

    temp = df[df["hit_group"].isin(group_order)].copy()
    if temp.empty:
        logging.warning("热度组数据为空，跳过箱线图。")
        return
    logging.info("hit_group 样本分布: %s", temp["hit_group"].value_counts().reindex(group_order, fill_value=0).to_dict())

    melted = temp.melt(id_vars=["hit_group"], value_vars=feats, var_name="feature", value_name="value")
    melted["feature_cn"] = melted["feature"].map(feat_label).fillna(melted["feature"])

    # 主图：按特征标准化，避免 tempo 量纲过大导致其他箱体被压扁
    eps = 1e-9
    melted["value_std"] = melted.groupby("feature")["value"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + eps))
    feat_order_cn = [feat_label.get(f, f) for f in feats]

    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=melted, x="feature_cn", y="value_std",
        hue="hit_group", order=feat_order_cn, hue_order=group_order,
        palette=palette, showfliers=False, width=0.72
    )
    plt.title("不同热度组音频特征对比（特征内标准化）")
    plt.xlabel("特征")
    plt.ylabel("标准化值（Z 分数）")
    plt.xticks(rotation=15)
    plt.legend(title="热度组", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(out_path)

    # 补充图：分面展示原始量纲，便于论文附录或正文解释
    n_feats = len(feats)
    n_cols = 3
    n_rows = math.ceil(n_feats / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.2 * n_rows), squeeze=False)
    for idx, feat in enumerate(feats):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        sns.boxplot(
            data=temp, x="hit_group", y=feat,
            hue="hit_group", order=group_order, hue_order=group_order,
            palette=palette, dodge=False,
            showfliers=False, ax=ax
        )
        # 分面图中颜色用于区分分组，图例重复信息较多，直接移除
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title(f"{feat_label.get(feat, feat)}（原始量纲）")
        ax.set_xlabel("热度组")
        ax.set_ylabel("取值")
        ax.tick_params(axis="x", rotation=12)

    for idx in range(n_feats, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.suptitle("不同热度组音频特征分面箱线图（原始量纲）", y=1.02, fontsize=16, fontweight="bold")
    fig.tight_layout()
    facet_path = out_path.with_name(f"{out_path.stem}_by_feature{out_path.suffix}")
    fig.savefig(facet_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("已保存图像: %s", facet_path)


def plot_spotify_violin_by_tier(df: pd.DataFrame, out_path: Path) -> None:
    feats = [c for c in ["danceability", "energy", "valence"] if c in df.columns]
    temp = df.melt(id_vars=["popularity_tier"], value_vars=feats, var_name="feature", value_name="value")
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=temp, x="feature", y="value", hue="popularity_tier", split=False, palette="Set2")
    plt.title("不同热度层级的特征分布")
    plt.xlabel("特征")
    plt.ylabel("取值")
    plt.legend(title="热度层级", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(out_path)


def plot_spotify_corr_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    feats = get_feature_columns_spotify(df)
    corr = df[feats + ["popularity"]].corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, square=True, linewidths=0.5)
    plt.title("Spotify 音频特征相关性热力图")
    save_figure(out_path)


def plot_spotify_duration_vs_popularity(df: pd.DataFrame, out_path: Path) -> None:
    if "duration_min" not in df.columns:
        return
    sample = df.sample(min(5000, len(df)), random_state=42)
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=sample, x="duration_min", y="popularity", hue="track_genre", alpha=0.45, s=40, legend=False)
    sns.regplot(data=sample, x="duration_min", y="popularity", scatter=False, color="#222222", line_kws={"lw": 2})
    plt.title("歌曲时长与热度关系")
    plt.xlabel("时长（分钟）")
    plt.ylabel("热度")
    save_figure(out_path)


def plot_spotify_pair_density(df: pd.DataFrame, out_path: Path) -> None:
    sample = df.sample(min(4000, len(df)), random_state=42)
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=sample, x="danceability", y="energy", fill=True, cmap="mako", thresh=0.05, levels=80)
    plt.title("舞动性与能量密度分布")
    plt.xlabel("舞动性")
    plt.ylabel("能量")
    save_figure(out_path)


def plot_spotify_genre_radar(df: pd.DataFrame, out_path: Path, top_n: int = 5) -> None:
    feats = [c for c in ["danceability", "energy", "speechiness", "acousticness", "liveness", "valence"] if c in df.columns]
    genres = df["track_genre"].value_counts().head(top_n).index.tolist()
    temp = df[df["track_genre"].isin(genres)].groupby("track_genre")[feats].mean()

    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    for i, (genre, row) in enumerate(temp.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, values, linewidth=2.2, label=genre, color=PALETTE_MAIN[i % len(PALETTE_MAIN)])
        ax.fill(angles, values, alpha=0.08, color=PALETTE_MAIN[i % len(PALETTE_MAIN)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feats)
    ax.set_title("流派音频画像雷达图", pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logging.info(f"已保存图像: {out_path}")


def plot_spotify_stacked_tier_by_genre(df: pd.DataFrame, out_path: Path, top_n: int = 8) -> None:
    top_genres = df["track_genre"].value_counts().head(top_n).index.tolist()
    temp = df[df["track_genre"].isin(top_genres)].copy()
    tab = pd.crosstab(temp["track_genre"], temp["popularity_tier"], normalize="index")
    tab = tab[[c for c in ["Low", "Mid-Low", "Mid-High", "High"] if c in tab.columns]]
    tab.plot(kind="bar", stacked=True, figsize=(12, 8), color=PALETTE_SOFT[:len(tab.columns)])
    plt.title("各流派热度层级构成")
    plt.xlabel("流派")
    plt.ylabel("占比")
    plt.legend(title="热度层级", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(out_path)


def plot_spotify_feature_map_pca(df: pd.DataFrame, out_path: Path) -> None:
    """有趣探索图：在 2D 特征空间中观察不同热度组的分布。"""
    if PCA is None or StandardScaler is None:
        logging.warning("未安装 scikit-learn，跳过 Spotify PCA 特征地图。")
        return

    feats = [c for c in ["danceability", "energy", "valence", "acousticness", "speechiness", "tempo", "popularity"] if c in df.columns]
    temp = df.dropna(subset=feats + ["hit_group"]).copy()
    if temp.empty:
        return

    temp = temp.sample(min(12000, len(temp)), random_state=42)
    x = StandardScaler().fit_transform(temp[feats])
    coords = PCA(n_components=2, random_state=42).fit_transform(x)
    temp["pc1"] = coords[:, 0]
    temp["pc2"] = coords[:, 1]

    palette = {"Hit": "#E15759", "Middle": "#4E79A7", "Low-Impact": "#59A14F"}
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=temp,
        x="pc1",
        y="pc2",
        hue="hit_group",
        palette=palette,
        alpha=0.32,
        s=25,
        linewidth=0
    )

    cent = temp.groupby("hit_group", as_index=False)[["pc1", "pc2"]].mean()
    plt.scatter(cent["pc1"], cent["pc2"], s=260, c="black", marker="*", label="Group Centroid")
    for _, row in cent.iterrows():
        plt.text(row["pc1"] + 0.08, row["pc2"] + 0.08, str(row["hit_group"]), fontsize=10, weight="bold")

    plt.title("Spotify 特征地图（PCA 二维分布）")
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.legend(title="热度分组", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(out_path)


# =========================================================
# 8. Billboard 数据静态图
# =========================================================
def plot_billboard_decade_trend(df: pd.DataFrame, out_path: Path) -> None:
    feats = [c for c in ["energy", "danceability", "happiness", "acousticness", "bpm"] if c in df.columns]
    temp = df.groupby("decade", as_index=False)[feats].mean().melt(id_vars="decade", var_name="feature", value_name="avg_value")
    plt.figure(figsize=(14, 8))
    palette = PALETTE_MAIN[: max(1, temp["feature"].nunique())]
    sns.lineplot(data=temp, x="decade", y="avg_value", hue="feature", marker="o", linewidth=2.5, palette=palette)
    plt.title("Billboard 冠军歌曲特征的年代演化")
    plt.xlabel("年代")
    plt.ylabel("平均值")
    save_figure(out_path)


def plot_billboard_weeks_distribution(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(12, 7))
    sns.histplot(df["weeks_at_number_one"].dropna(), bins=20, kde=True, color=PALETTE_MAIN[2], edgecolor="white")
    plt.title("冠军周数分布")
    plt.xlabel("冠军周数")
    plt.ylabel("数量")
    save_figure(out_path)


def plot_billboard_top_artists(df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    temp = df["artist"].value_counts().head(top_n).sort_values(ascending=True).reset_index()
    temp.columns = ["artist", "count"]
    plt.figure(figsize=(12, 9))
    sns.barplot(data=temp, x="count", y="artist", color=PALETTE_MAIN[2])
    plt.title(f"冠军歌曲数量前 {top_n} 的艺人")
    plt.xlabel("冠军歌曲数量")
    plt.ylabel("艺人")
    save_figure(out_path)


def plot_billboard_topic_distribution(df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    if "lyrical_topic" not in df.columns:
        return
    temp = df["lyrical_topic"].fillna("Unknown").str.split(";").explode().str.strip()
    temp = temp[temp.notna() & (temp != "")]
    temp = temp.value_counts().head(top_n).sort_values(ascending=True).reset_index()
    temp.columns = ["topic", "count"]
    plt.figure(figsize=(12, 8))
    sns.barplot(data=temp, x="count", y="topic", color=PALETTE_SOFT[2])
    plt.title("Billboard 冠军歌曲高频歌词主题")
    plt.xlabel("数量")
    plt.ylabel("主题")
    save_figure(out_path)


def plot_billboard_bubble(df: pd.DataFrame, out_path: Path) -> None:
    sample = df.dropna(subset=["year", "danceability", "energy", "weeks_at_number_one"]).copy()
    plt.figure(figsize=(13, 8))
    sizes = np.clip(sample["weeks_at_number_one"].fillna(1) * 25, 40, 400)
    plt.scatter(sample["danceability"], sample["energy"], s=sizes, c=sample["year"], cmap="viridis", alpha=0.65, edgecolors="white", linewidth=0.5)
    cbar = plt.colorbar()
    cbar.set_label("Year")
    plt.title("Billboard 冠军歌曲：舞动性与能量（气泡=冠军周数）")
    plt.xlabel("舞动性")
    plt.ylabel("能量")
    save_figure(out_path)


def plot_billboard_corr_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    feats = get_feature_columns_billboard(df)
    corr = df[feats + ["weeks_at_number_one"]].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Spectral", center=0, square=True, linewidths=0.5)
    plt.title("Billboard 冠军歌曲特征相关性热力图")
    save_figure(out_path)


def extract_billboard_topics_long(billboard_df: pd.DataFrame, topic_catalog_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """将分号分隔的 lyrical_topic 展开成长表，用于热力图与共现图。"""
    valid_topics = set()
    if topic_catalog_df is not None and "lyrical_topics" in topic_catalog_df.columns:
        valid_topics = set(topic_catalog_df["lyrical_topics"].dropna().astype(str).str.strip())

    rows = []
    for idx, row in billboard_df.iterrows():
        raw = row.get("lyrical_topic")
        if pd.isna(raw):
            continue
        topics = [x.strip() for x in str(raw).split(";") if x and x.strip() and x.strip().lower() != "nan"]
        for tp in topics:
            if valid_topics and tp not in valid_topics:
                continue
            rows.append(
                {
                    "song_idx": idx,
                    "song": row.get("song"),
                    "artist": row.get("artist"),
                    "decade": row.get("decade"),
                    "topic": tp
                }
            )
    return pd.DataFrame(rows)


def plot_billboard_topic_decade_heatmap(topics_long_df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    if topics_long_df.empty:
        logging.warning("主题长表为空，跳过 decade-topic 热力图。")
        return
    top_topics = topics_long_df["topic"].value_counts().head(top_n).index.tolist()
    temp = topics_long_df[topics_long_df["topic"].isin(top_topics)].copy()
    tab = pd.crosstab(temp["decade"], temp["topic"], normalize="index").sort_index()

    plt.figure(figsize=(14, 8))
    sns.heatmap(tab, cmap="YlGnBu", linewidths=0.4, linecolor="white")
    plt.title("各年代主题构成（按行归一化）")
    plt.xlabel("主题")
    plt.ylabel("年代")
    save_figure(out_path)


def plot_billboard_topic_cooccurrence(topics_long_df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    if topics_long_df.empty:
        logging.warning("主题长表为空，跳过主题共现热力图。")
        return

    top_topics = topics_long_df["topic"].value_counts().head(top_n).index.tolist()
    temp = topics_long_df[topics_long_df["topic"].isin(top_topics)].copy()
    by_song = temp.groupby("song_idx")["topic"].apply(lambda s: sorted(set(s.dropna().tolist())))

    topic_to_i = {t: i for i, t in enumerate(top_topics)}
    mat = np.zeros((len(top_topics), len(top_topics)), dtype=np.int32)
    for t_list in by_song:
        for i in range(len(t_list)):
            for j in range(i, len(t_list)):
                a, b = t_list[i], t_list[j]
                ia, ib = topic_to_i[a], topic_to_i[b]
                mat[ia, ib] += 1
                if ia != ib:
                    mat[ib, ia] += 1

    co_df = pd.DataFrame(mat, index=top_topics, columns=top_topics)
    # 去掉对角线自共现，避免色阶被自身频次主导
    co_only = co_df.copy()
    for t in co_only.index:
        co_only.loc[t, t] = 0

    if int((co_only.values > 0).sum()) == 0:
        logging.warning("主题两两共现均为 0，保留原矩阵显示。")
        plot_df = co_df
        annot_df = co_df.astype(str)
        title = "主题共现矩阵（同一冠军歌曲内）"
    else:
        # 使用 log1p 放大低频共现细节
        plot_df = np.log1p(co_only)
        annot_df = co_only.astype(int).astype(str)
        annot_df[co_only == 0] = ""
        title = "主题共现矩阵（去除对角自共现，色阶为 log1p）"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        plot_df, cmap="mako", linewidths=0.3, linecolor="white",
        annot=annot_df, fmt="", cbar_kws={"label": "共现强度（log1p）"}
    )
    plt.title(title)
    plt.xlabel("主题")
    plt.ylabel("主题")
    save_figure(out_path)


def plot_billboard_topic_network(topics_long_df: pd.DataFrame, out_path: Path, top_n: int = 14, min_edge_count: int = 6) -> None:
    """主题共现网络图：节点为主题，边表示在同一冠军歌曲中共现。"""
    if topics_long_df.empty:
        logging.warning("主题长表为空，跳过主题共现网络图。")
        return

    top_topics = topics_long_df["topic"].value_counts().head(top_n).index.tolist()
    temp = topics_long_df[topics_long_df["topic"].isin(top_topics)].copy()
    if temp.empty:
        logging.warning("主题筛选后为空，跳过主题共现网络图。")
        return

    by_song = temp.groupby("song_idx")["topic"].apply(lambda s: sorted(set(s.dropna().tolist())))
    topic_to_i = {t: i for i, t in enumerate(top_topics)}
    mat = np.zeros((len(top_topics), len(top_topics)), dtype=np.int32)
    for t_list in by_song:
        for i in range(len(t_list)):
            for j in range(i + 1, len(t_list)):
                a, b = t_list[i], t_list[j]
                ia, ib = topic_to_i[a], topic_to_i[b]
                mat[ia, ib] += 1
                mat[ib, ia] += 1

    deg = mat.sum(axis=1)
    max_deg = max(float(deg.max()), 1.0)
    node_sizes = 600 + 2800 * (deg / max_deg)

    # 使用圆环布局，减少依赖，保证稳定生成
    angles = np.linspace(0, 2 * np.pi, len(top_topics), endpoint=False)
    radius = 1.0
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_facecolor("#FAFAFA")

    max_w = max(float(mat.max()), 1.0)
    for i in range(len(top_topics)):
        for j in range(i + 1, len(top_topics)):
            w = mat[i, j]
            if w < min_edge_count:
                continue
            alpha = min(0.75, 0.15 + 0.6 * (w / max_w))
            lw = 0.4 + 4.0 * (w / max_w)
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#6B7280", linewidth=lw, alpha=alpha, zorder=1)

    topic_color_map = build_topic_color_map(top_topics)
    for i, topic in enumerate(top_topics):
        node_color = topic_color_map.get(topic, TOPIC_COLOR_FALLBACK[i % len(TOPIC_COLOR_FALLBACK)])
        ax.scatter(xs[i], ys[i], s=float(node_sizes[i]), color=node_color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(xs[i], ys[i], topic, ha="center", va="center", fontsize=9, weight="bold", zorder=4)

    ax.set_title("歌词主题共现网络图（Billboard 冠军歌曲）", fontsize=17, fontweight="bold")
    ax.text(
        0.5, -0.08,
        "说明：边越粗表示主题在同一首冠军歌曲中共现次数越多（探索性关联，非因果结论）",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#374151"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    for s in ax.spines.values():
        s.set_visible(False)
    save_figure(out_path)


def plot_billboard_decade_topic_sankey(topics_long_df: pd.DataFrame, out_path: Path, top_n_topics: int = 12) -> None:
    """年代到主题的迁移桑基图（HTML）。"""
    if topics_long_df.empty:
        logging.warning("主题长表为空，跳过年代-主题桑基图。")
        return

    temp = topics_long_df.dropna(subset=["decade", "topic"]).copy()
    if temp.empty:
        logging.warning("缺少 decade/topic 有效记录，跳过年代-主题桑基图。")
        return

    top_topics = temp["topic"].value_counts().head(top_n_topics).index.tolist()
    temp = temp[temp["topic"].isin(top_topics)].copy()
    if temp.empty:
        logging.warning("主题筛选后为空，跳过年代-主题桑基图。")
        return

    temp["decade"] = pd.to_numeric(temp["decade"], errors="coerce")
    temp = temp.dropna(subset=["decade"])
    temp["decade"] = temp["decade"].astype(int)
    if temp.empty:
        return

    flow = temp.groupby(["decade", "topic"], as_index=False).size().rename(columns={"size": "count"})
    decade_labels = [f"{d}年代" for d in sorted(flow["decade"].unique().tolist())]
    topic_labels = [f"主题：{t}" for t in top_topics]
    nodes = decade_labels + topic_labels

    node_index = {name: i for i, name in enumerate(nodes)}
    sources = [node_index[f"{int(row.decade)}年代"] for _, row in flow.iterrows()]
    targets = [node_index[f"主题：{row.topic}"] for _, row in flow.iterrows()]
    values = flow["count"].astype(float).tolist()

    topic_color_map = build_topic_color_map(top_topics)

    decade_node_color = "rgba(59,130,246,0.75)"
    topic_node_colors = [hex_to_rgba(topic_color_map[t], 0.85) for t in top_topics]
    node_colors = [decade_node_color] * len(decade_labels) + topic_node_colors

    link_colors = [
        hex_to_rgba(topic_color_map[row.topic], 0.40) for _, row in flow.iterrows()
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=nodes,
                    color=node_colors,
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(255,255,255,0.75)", width=0.6)
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors
                )
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title="Billboard 冠军歌曲：年代到歌词主题的流向（桑基图）",
        font=PLOTLY_CHART_FONT,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logging.info("已保存交互图: %s", out_path)


def plot_topic_color_legend(topics_long_df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    """输出主题-颜色固定映射图例。"""
    if topics_long_df.empty:
        logging.warning("主题长表为空，跳过主题颜色图例。")
        return

    top_topics = topics_long_df["topic"].value_counts().head(top_n).index.tolist()
    if not top_topics:
        return
    topic_color_map = build_topic_color_map(top_topics)

    n = len(top_topics)
    n_cols = 2 if n <= 14 else 3
    n_rows = math.ceil(n / n_cols)
    fig_h = max(4.6, n_rows * 0.92 + 1.3)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.set_facecolor("#F8FAFC")

    x_gap = 1.0 / n_cols
    y_step = 1.0 / max(n_rows + 0.5, 1.0)
    box_w = 0.028
    box_h = min(0.06, y_step * 0.6)

    for idx, topic in enumerate(top_topics):
        c = idx // n_rows
        r = idx % n_rows
        x0 = 0.04 + c * x_gap
        y0 = 0.96 - (r + 1) * y_step
        color = topic_color_map.get(topic, TOPIC_COLOR_FALLBACK[idx % len(TOPIC_COLOR_FALLBACK)])
        rect = plt.Rectangle((x0, y0), box_w, box_h, color=color, transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(
            x0 + box_w + 0.012, y0 + box_h / 2, topic,
            transform=ax.transAxes, va="center", ha="left", fontsize=11, color="#111827"
        )

    ax.set_title("主题-颜色固定映射图例", fontsize=16, fontweight="bold", pad=12)
    ax.text(
        0.5, 0.01, "说明：该映射在桑基图与主题网络图中保持一致", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=10, color="#475569"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("已保存图像: %s", out_path)


def plot_billboard_decade_style_trajectory(billboard_df: pd.DataFrame, out_path: Path) -> None:
    """1) 年代风格轨迹图：展示冠军歌曲音频画像在二维空间中的年代迁移。"""
    feat_cols = [c for c in ["energy", "danceability", "happiness", "acousticness", "bpm", "loudness_d_b"] if c in billboard_df.columns]
    if "decade" not in billboard_df.columns or len(feat_cols) < 3:
        logging.warning("缺少 decade 或特征列不足，跳过年代风格轨迹图。")
        return

    temp = billboard_df.dropna(subset=["decade"] + feat_cols).copy()
    if len(temp) < 30:
        logging.warning("可用样本过少，跳过年代风格轨迹图。")
        return
    temp["decade"] = pd.to_numeric(temp["decade"], errors="coerce")
    temp = temp.dropna(subset=["decade"])
    temp["decade"] = temp["decade"].astype(int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(temp[feat_cols])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)
    temp["pc1"] = coords[:, 0]
    temp["pc2"] = coords[:, 1]

    centers = (
        temp.groupby("decade", as_index=False)
        .agg(pc1=("pc1", "mean"), pc2=("pc2", "mean"), n=("pc1", "size"))
        .sort_values("decade")
    )
    if centers.empty or len(centers) < 2:
        return

    plt.figure(figsize=(13, 8.5))
    ax = plt.gca()
    ax.set_facecolor("#F8FAFC")

    # 背景散点：展示原始样本分布
    bg = temp.sample(min(len(temp), 900), random_state=42)
    plt.scatter(bg["pc1"], bg["pc2"], s=18, color="#94A3B8", alpha=0.18, edgecolors="none", zorder=1)

    cmap = plt.cm.viridis
    decade_vals = centers["decade"].tolist()
    n_dec = len(decade_vals)
    colors = [cmap(i / max(n_dec - 1, 1)) for i in range(n_dec)]

    # 轨迹箭头
    for i in range(n_dec - 1):
        x0, y0 = float(centers.iloc[i]["pc1"]), float(centers.iloc[i]["pc2"])
        x1, y1 = float(centers.iloc[i + 1]["pc1"]), float(centers.iloc[i + 1]["pc2"])
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#334155", alpha=0.85),
            zorder=3
        )

    # 年代节点
    n_min, n_max = centers["n"].min(), centers["n"].max()
    denom = max(float(n_max - n_min), 1.0)
    for i, row in centers.iterrows():
        size = 120 + 360 * (float(row["n"]) - float(n_min)) / denom
        plt.scatter(row["pc1"], row["pc2"], s=size, color=colors[i], edgecolor="white", linewidth=1.2, zorder=4)
        plt.text(
            float(row["pc1"]) + 0.06, float(row["pc2"]) + 0.06, f"{int(row['decade'])}年代",
            fontsize=10, weight="bold", color="#0F172A", zorder=5
        )

    explained = pca.explained_variance_ratio_
    plt.title("Billboard 冠军歌曲年代风格轨迹图（PCA）")
    plt.xlabel(f"主成分 1（解释方差 {explained[0]:.1%}）")
    plt.ylabel(f"主成分 2（解释方差 {explained[1]:.1%}）")
    plt.grid(alpha=0.18)
    save_figure(out_path)


def plot_billboard_partial_corr_network_by_decade(
    billboard_df: pd.DataFrame, out_path: Path, min_abs_corr: float = 0.16
) -> None:
    """3) 控制年代后的相关网络图：先剔除 decade 线性趋势，再看特征关联。"""
    feat_cols = [c for c in ["energy", "danceability", "happiness", "acousticness", "bpm", "loudness_d_b"] if c in billboard_df.columns]
    if "decade" not in billboard_df.columns or len(feat_cols) < 4:
        logging.warning("缺少 decade 或特征列不足，跳过条件相关网络图。")
        return

    temp = billboard_df.dropna(subset=["decade"] + feat_cols).copy()
    if len(temp) < 40:
        return
    temp["decade"] = pd.to_numeric(temp["decade"], errors="coerce")
    temp = temp.dropna(subset=["decade"])
    if temp.empty:
        return

    x = temp["decade"].astype(float).to_numpy()
    resid = pd.DataFrame(index=temp.index)
    for feat in feat_cols:
        y = temp[feat].astype(float).to_numpy()
        if len(np.unique(x)) >= 2:
            a, b = np.polyfit(x, y, 1)
            resid[feat] = y - (a * x + b)
        else:
            resid[feat] = y - y.mean()

    corr = resid.corr().fillna(0.0)
    np.fill_diagonal(corr.values, 0.0)

    # 圆环布局
    n = len(feat_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)
    feat_to_xy = {feat_cols[i]: (xs[i], ys[i]) for i in range(n)}
    feat_label = {
        "energy": "能量", "danceability": "舞动性", "happiness": "愉悦度",
        "acousticness": "原声占比", "bpm": "节奏速度", "loudness_d_b": "响度"
    }

    plt.figure(figsize=(11.8, 11.2))
    ax = plt.gca()
    ax.set_facecolor("#F8FAFC")

    max_abs = max(float(np.abs(corr.values).max()), min_abs_corr)
    edge_count = 0
    for i, a in enumerate(feat_cols):
        for j, b in enumerate(feat_cols):
            if j <= i:
                continue
            w = float(corr.loc[a, b])
            if abs(w) < min_abs_corr:
                continue
            edge_count += 1
            lw = 1.0 + 5.0 * abs(w) / max_abs
            color = "#DC2626" if w > 0 else "#2563EB"
            alpha = min(0.88, 0.28 + 0.56 * abs(w) / max_abs)
            xa, ya = feat_to_xy[a]
            xb, yb = feat_to_xy[b]
            ax.plot([xa, xb], [ya, yb], color=color, linewidth=lw, alpha=alpha, zorder=1)

    strength = np.abs(corr).sum(axis=1).reindex(feat_cols)
    s_min, s_max = float(strength.min()), float(strength.max())
    s_den = max(s_max - s_min, 1e-8)
    for i, feat in enumerate(feat_cols):
        x0, y0 = feat_to_xy[feat]
        node_size = 900 + 1800 * (float(strength.loc[feat]) - s_min) / s_den
        node_color = PALETTE_MAIN[i % len(PALETTE_MAIN)]
        ax.scatter(x0, y0, s=node_size, color=node_color, edgecolor="white", linewidth=1.6, zorder=3)
        ax.text(x0, y0, feat_label.get(feat, feat), ha="center", va="center", fontsize=10, weight="bold", zorder=4)

    ax.plot([], [], color="#DC2626", lw=2.4, label="正相关（控制年代后）")
    ax.plot([], [], color="#2563EB", lw=2.4, label="负相关（控制年代后）")
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.5, -0.06, f"仅展示 |相关系数| ≥ {min_abs_corr:.2f} 的边，共 {edge_count} 条",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#475569"
    )
    ax.set_title("控制年代后的音频特征相关网络图（Billboard）", fontsize=17, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    for s in ax.spines.values():
        s.set_visible(False)
    save_figure(out_path)


def plot_billboard_topic_community_graph(
    topics_long_df: pd.DataFrame, out_path: Path, top_n: int = 16, min_edge_count: int = 4, n_clusters: int = 4
) -> None:
    """7) 主题共现社区图：对主题共现网络做社区划分并可视化。"""
    if topics_long_df.empty:
        logging.warning("主题长表为空，跳过主题共现社区图。")
        return

    top_topics = topics_long_df["topic"].value_counts().head(top_n).index.tolist()
    temp = topics_long_df[topics_long_df["topic"].isin(top_topics)].copy()
    if temp.empty or len(top_topics) < 4:
        return

    by_song = temp.groupby("song_idx")["topic"].apply(lambda s: sorted(set(s.dropna().tolist())))
    topic_to_i = {t: i for i, t in enumerate(top_topics)}
    mat = np.zeros((len(top_topics), len(top_topics)), dtype=np.int32)
    for t_list in by_song:
        for i in range(len(t_list)):
            for j in range(i + 1, len(t_list)):
                a, b = t_list[i], t_list[j]
                ia, ib = topic_to_i[a], topic_to_i[b]
                mat[ia, ib] += 1
                mat[ib, ia] += 1

    np.fill_diagonal(mat, 0)
    if int((mat > 0).sum()) == 0:
        logging.warning("主题间共现全为 0，跳过主题共现社区图。")
        return

    affinity = mat.astype(float)
    max_v = max(float(affinity.max()), 1.0)
    affinity = affinity / max_v

    k = max(2, min(n_clusters, len(top_topics) - 1))
    try:
        sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=42, assign_labels="kmeans")
        labels = sc.fit_predict(affinity)
    except Exception as exc:
        logging.warning("谱聚类失败，改用度排序分组: %s", exc)
        deg = mat.sum(axis=1)
        order = np.argsort(-deg)
        labels = np.zeros(len(top_topics), dtype=int)
        for rank, idx in enumerate(order):
            labels[idx] = rank % k

    # 用 PCA(邻接向量)做稳定布局
    try:
        pos = PCA(n_components=2, random_state=42).fit_transform(affinity)
    except Exception:
        ang = np.linspace(0, 2 * np.pi, len(top_topics), endpoint=False)
        pos = np.column_stack([np.cos(ang), np.sin(ang)])
    if np.allclose(np.std(pos, axis=0), 0):
        ang = np.linspace(0, 2 * np.pi, len(top_topics), endpoint=False)
        pos = np.column_stack([np.cos(ang), np.sin(ang)])

    # 归一化坐标，增强观感
    pos = pos - pos.mean(axis=0, keepdims=True)
    std = np.std(pos, axis=0, keepdims=True)
    std[std == 0] = 1.0
    pos = pos / std

    comm_palette = ["#FF6B6B", "#4ECDC4", "#FFD166", "#06D6A0", "#3A86FF", "#8338EC"]
    node_colors = [comm_palette[int(lb) % len(comm_palette)] for lb in labels]
    deg = mat.sum(axis=1).astype(float)
    d_min, d_max = float(deg.min()), float(deg.max())
    d_den = max(d_max - d_min, 1e-8)

    plt.figure(figsize=(12.8, 10.6))
    ax = plt.gca()
    ax.set_facecolor("#F8FAFC")

    # 先画边
    max_w = max(float(mat.max()), 1.0)
    edge_num = 0
    for i in range(len(top_topics)):
        for j in range(i + 1, len(top_topics)):
            w = int(mat[i, j])
            if w < min_edge_count:
                continue
            edge_num += 1
            alpha = min(0.72, 0.14 + 0.58 * w / max_w)
            lw = 0.5 + 3.2 * w / max_w
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="#64748B", linewidth=lw, alpha=alpha, zorder=1)

    # 再画点与标签
    for i, topic in enumerate(top_topics):
        size = 700 + 2300 * (deg[i] - d_min) / d_den
        ax.scatter(pos[i, 0], pos[i, 1], s=size, color=node_colors[i], edgecolor="white", linewidth=1.3, zorder=3)
        ax.text(pos[i, 0], pos[i, 1], topic, ha="center", va="center", fontsize=9.6, weight="bold", zorder=4)

    # 社区图例
    k_show = int(np.max(labels)) + 1
    for c in range(k_show):
        ax.scatter([], [], s=180, color=comm_palette[c % len(comm_palette)], label=f"社区 {c + 1}")
    ax.legend(loc="upper left", frameon=False, ncol=2)

    ax.text(
        0.5, -0.06, f"共现边阈值: ≥ {min_edge_count}，可视化边数: {edge_num}",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#475569"
    )
    ax.set_title("歌词主题共现社区图（Billboard 冠军歌曲）", fontsize=17, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save_figure(out_path)


def plot_billboard_feature_ridgeline_by_decade(
    billboard_df: pd.DataFrame, out_path: Path, min_count: int = 20
) -> None:
    """按年代展示关键音频特征分布的脊线图。"""
    feats = [c for c in ["energy", "danceability", "happiness"] if c in billboard_df.columns]
    if "decade" not in billboard_df.columns or not feats:
        logging.warning("缺少 decade 或关键特征列，跳过脊线图。")
        return

    temp = billboard_df.dropna(subset=["decade"]).copy()
    temp["decade"] = pd.to_numeric(temp["decade"], errors="coerce")
    temp = temp.dropna(subset=["decade"])
    if temp.empty:
        return
    temp["decade"] = temp["decade"].astype(int)
    decades = sorted(temp["decade"].unique().tolist())

    feat_label = {"energy": "能量", "danceability": "舞动性", "happiness": "愉悦度"}
    fig, axes = plt.subplots(len(feats), 1, figsize=(13, 4.2 * len(feats)), sharex=False)
    if len(feats) == 1:
        axes = [axes]

    for ax, feat in zip(axes, feats):
        col = temp[feat].dropna()
        if col.empty:
            ax.set_visible(False)
            continue

        lo, hi = np.nanpercentile(col, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(col.min()), float(col.max())
        bins = np.linspace(lo, hi, 140)
        centers = (bins[:-1] + bins[1:]) / 2
        kernel = np.array([1, 4, 6, 4, 1], dtype=float)
        kernel = kernel / kernel.sum()

        used = 0
        for i, dec in enumerate(decades):
            vals = temp.loc[temp["decade"] == dec, feat].dropna().to_numpy()
            if len(vals) < min_count:
                continue
            hist, _ = np.histogram(vals, bins=bins, density=True)
            smooth = np.convolve(hist, kernel, mode="same")
            if float(smooth.max()) <= 0:
                continue
            y0 = used
            y1 = y0 + (smooth / smooth.max()) * 0.9
            color = PALETTE_MAIN[i % len(PALETTE_MAIN)]
            ax.fill_between(centers, y0, y1, color=color, alpha=0.62, linewidth=0)
            ax.plot(centers, y1, color=color, linewidth=1.4)
            ax.text(centers.min(), y0 + 0.06, f"{dec}年代", fontsize=9, color="#111827")
            used += 1

        ax.set_title(f"{feat_label.get(feat, feat)}分布的年代脊线图")
        ax.set_xlabel("特征取值")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.15)
        for s in ["left", "right", "top"]:
            ax.spines[s].set_visible(False)

    fig.suptitle("Billboard 冠军歌曲关键特征的年代分布脊线图", y=1.02, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logging.info("已保存图像: %s", out_path)


def plot_bootstrap_ci_hit_vs_low(
    spotify_df: pd.DataFrame, out_path: Path, n_boot: int = 1200, sample_n: int = 3000
) -> None:
    """Bootstrap 估计 Hit 与 Low-Impact 在关键特征上的均值差及置信区间。"""
    feats = [c for c in ["danceability", "energy", "valence", "acousticness", "tempo"] if c in spotify_df.columns]
    if "hit_group" not in spotify_df.columns or not feats:
        return

    hit = spotify_df[spotify_df["hit_group"] == "Hit"]
    low = spotify_df[spotify_df["hit_group"] == "Low-Impact"]
    if hit.empty or low.empty:
        logging.warning("Hit 或 Low-Impact 分组为空，跳过 Bootstrap 置信区间图。")
        return

    feat_label = {
        "danceability": "舞动性",
        "energy": "能量",
        "valence": "愉悦度",
        "acousticness": "原声占比",
        "tempo": "节奏速度"
    }
    rng = np.random.default_rng(42)
    rows = []

    for feat in feats:
        a = hit[feat].dropna().to_numpy()
        b = low[feat].dropna().to_numpy()
        if len(a) < 10 or len(b) < 10:
            continue
        n = min(sample_n, len(a), len(b))

        diffs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            a_mean = rng.choice(a, size=n, replace=True).mean()
            b_mean = rng.choice(b, size=n, replace=True).mean()
            diffs[i] = a_mean - b_mean

        rows.append(
            {
                "feature": feat,
                "feature_cn": feat_label.get(feat, feat),
                "obs_diff": float(a.mean() - b.mean()),
                "ci_low": float(np.percentile(diffs, 2.5)),
                "ci_high": float(np.percentile(diffs, 97.5)),
            }
        )

    if not rows:
        return

    out = pd.DataFrame(rows).sort_values("obs_diff")
    y = np.arange(len(out))

    plt.figure(figsize=(12, 7))
    plt.hlines(y=y, xmin=out["ci_low"], xmax=out["ci_high"], color="#6B7280", linewidth=2.5, alpha=0.85)
    plt.scatter(out["obs_diff"], y, color="#DC2626", s=80, zorder=3, label="观测均值差")
    plt.axvline(0, color="#111827", linestyle="--", linewidth=1.4, alpha=0.85)
    plt.yticks(y, out["feature_cn"])
    plt.xlabel("均值差（Hit - Low-Impact）")
    plt.ylabel("特征")
    plt.title("Hit 与 Low-Impact 的特征差异（Bootstrap 95% 置信区间）")
    plt.legend(loc="lower right")
    save_figure(out_path)


# =========================================================
# 9. 对照分析图
# =========================================================
def _billboard_metric_to_spotify_scale(series: pd.Series, tempo_like: bool) -> pd.Series:
    """将 Billboard 侧比例类指标对齐到 Spotify 的 0–1；tempo/bpm 保持 BPM 原样。"""
    v = pd.to_numeric(series, errors="coerce")
    if tempo_like:
        return v
    finite = v.dropna()
    if finite.empty:
        return v
    if float(finite.max()) <= 1.2:
        return v.clip(lower=0.0, upper=1.0)
    return (v / 100.0).clip(lower=0.0, upper=1.0)


def build_comparison_table(spotify_df: pd.DataFrame, billboard_df: pd.DataFrame) -> pd.DataFrame:
    """构造 Spotify 爆款与 Billboard 冠军歌曲的特征对照表（Billboard 比例列已换算至 0–1 与 Spotify 可比）。"""
    sp_feats = [c for c in ["danceability", "energy", "valence", "acousticness", "tempo"] if c in spotify_df.columns]
    bb_map = {
        "danceability": "danceability",
        "energy": "energy",
        "valence": "happiness",          # 语义上近似，但不是完全等价，这里在报告中会说明。
        "acousticness": "acousticness",
        "tempo": "bpm"
    }

    hit_spotify = spotify_df[spotify_df["hit_group"] == "Hit"]
    rows = []
    for feat in sp_feats:
        bb_feat = bb_map.get(feat)
        if bb_feat in billboard_df.columns:
            bb_mean = float(
                _billboard_metric_to_spotify_scale(
                    billboard_df[bb_feat], tempo_like=(feat == "tempo")
                ).mean()
            )
            rows.append({
                "feature": feat,
                "spotify_hit_mean": float(hit_spotify[feat].mean()),
                "billboard_no1_mean": bb_mean,
            })
    comp = pd.DataFrame(rows)
    return comp


def plot_comparison_lollipop(comp_df: pd.DataFrame, out_path: Path) -> None:
    if comp_df.empty:
        return
    temp = comp_df.copy().sort_values("spotify_hit_mean")
    y = np.arange(len(temp))
    plt.figure(figsize=(12, 7))
    plt.hlines(y=y, xmin=temp["spotify_hit_mean"], xmax=temp["billboard_no1_mean"], color="#B0B0B0", linewidth=2)
    plt.scatter(temp["spotify_hit_mean"], y, color=PALETTE_MAIN[0], s=120, label="Spotify Hits")
    plt.scatter(temp["billboard_no1_mean"], y, color=PALETTE_MAIN[2], s=120, label="Billboard #1")
    plt.yticks(y, temp["feature"])
    plt.xlabel("特征平均值（比例类已统一为 0–1；tempo 为 BPM）")
    plt.title("Spotify 高热歌曲与 Billboard 冠军歌曲对比")
    plt.legend()
    save_figure(out_path)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / denom)


def plot_feature_gap_heatmap_by_decade(spotify_df: pd.DataFrame, billboard_df: pd.DataFrame, out_path: Path) -> None:
    """年代维度的差异热力图：Billboard 年代均值相对 Spotify 高热均值。"""
    feat_map = {
        "danceability": "danceability",
        "energy": "energy",
        "valence": "happiness",      # 概念近似，不等价
        "acousticness": "acousticness",
        "tempo": "bpm",
    }

    hit_spotify = spotify_df[spotify_df["hit_group"] == "Hit"]
    if hit_spotify.empty:
        return

    sp_means = {f: hit_spotify[f].mean() for f in feat_map if f in hit_spotify.columns}
    rows = []
    for decade, grp in billboard_df.groupby("decade"):
        row = {"decade": int(decade)}
        for sp_feat, bb_feat in feat_map.items():
            if sp_feat in sp_means and bb_feat in grp.columns:
                bb_mean = float(
                    _billboard_metric_to_spotify_scale(
                        grp[bb_feat], tempo_like=(sp_feat == "tempo")
                    ).mean()
                )
                row[sp_feat] = bb_mean - sp_means[sp_feat]
        rows.append(row)

    gap = pd.DataFrame(rows).set_index("decade").sort_index()
    if gap.empty:
        return

    plt.figure(figsize=(12, 7))
    sns.heatmap(gap.T, cmap="coolwarm", center=0, annot=True, fmt=".2f", linewidths=0.35, linecolor="white")
    plt.title("Billboard 年代均值相对 Spotify 高热均值的特征差值")
    plt.xlabel("年代")
    plt.ylabel("特征")
    save_figure(out_path)


def plot_decade_similarity_to_spotify_hits(spotify_df: pd.DataFrame, billboard_df: pd.DataFrame, out_path: Path) -> None:
    """看各年代 Billboard 与 Spotify 高热组特征轮廓的相似度。"""
    feat_map = {
        "danceability": "danceability",
        "energy": "energy",
        "valence": "happiness",      # 概念近似，不等价
        "acousticness": "acousticness",
        "tempo": "bpm",
    }
    hit_spotify = spotify_df[spotify_df["hit_group"] == "Hit"]
    usable = [k for k, v in feat_map.items() if k in hit_spotify.columns and v in billboard_df.columns]
    if not usable:
        return

    sp_vec = np.array([hit_spotify[k].mean() for k in usable], dtype=float)
    rows = []
    for decade, grp in billboard_df.groupby("decade"):
        bb_vec = np.array(
            [
                float(
                    _billboard_metric_to_spotify_scale(
                        grp[feat_map[k]], tempo_like=(k == "tempo")
                    ).mean()
                )
                for k in usable
            ],
            dtype=float,
        )
        rows.append({"decade": int(decade), "cosine_similarity": _cosine_similarity(sp_vec, bb_vec)})

    sim_df = pd.DataFrame(rows).sort_values("decade")
    plt.figure(figsize=(12, 6))
    # 使用数值型 x 轴，避免 seaborn 将年代刻度当成分类字符串时的 Matplotlib 提示（见《问题排查》4.1）。
    decades = sim_df["decade"].astype(int).to_numpy()
    plt.bar(decades, sim_df["cosine_similarity"].to_numpy(), width=8, color=PALETTE_MAIN[0], align="center")
    plt.xticks(decades)
    plt.ylim(-1, 1)
    plt.title("Billboard 各年代与 Spotify 高热画像相似度")
    plt.xlabel("年代")
    plt.ylabel("余弦相似度")
    save_figure(out_path)


# =========================================================
# 10. 交互式 3D 图
# =========================================================
def plot_interactive_3d_cloud(df: pd.DataFrame, out_path: Path) -> None:
    """交互式 3D 音乐云图（HTML）：用于论文展示探索性关联。"""

    req_cols = ["danceability", "energy", "valence", "popularity"]
    if not all(c in df.columns for c in req_cols):
        logging.warning("缺少必要列，跳过交互式 3D 图。")
        return

    temp = df.dropna(subset=req_cols).copy()
    if temp.empty:
        return

    # 控制交互页面体积与渲染流畅度
    temp = temp.sample(min(12000, len(temp)), random_state=42)

    if "track_genre" in temp.columns:
        top_genres = temp["track_genre"].value_counts().head(12).index
        temp["genre_top"] = np.where(temp["track_genre"].isin(top_genres), temp["track_genre"], "Other")
    else:
        temp["genre_top"] = "Unknown"

    pop_min = float(temp["popularity"].min())
    pop_max = float(temp["popularity"].max())
    span = max(pop_max - pop_min, 1e-6)
    temp["size_scaled"] = 4 + (temp["popularity"] - pop_min) / span * 10

    hover_cols = [c for c in ["track_name", "artists", "album_name", "track_genre", "popularity"] if c in temp.columns]
    fig = px.scatter_3d(
        temp,
        x="danceability",
        y="energy",
        z="valence",
        color="genre_top",
        size="size_scaled",
        opacity=0.70,
        hover_data=hover_cols,
        title="交互式 3D 音乐特征云图（探索性关联）",
        color_discrete_sequence=PALETTE_MAIN,
    )

    fig.update_traces(marker=dict(line=dict(width=0.2, color="rgba(255,255,255,0.55)")))
    fig.update_layout(
        template="plotly_white",
        font=PLOTLY_CHART_FONT,
        legend_title_text="流派（前12 + 其他）",
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title="舞动性",
            yaxis_title="能量",
            zaxis_title="愉悦度",
            bgcolor="rgba(245,247,250,1)",
            camera=dict(eye=dict(x=1.6, y=1.4, z=1.1)),
        ),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logging.info("已保存交互图: %s", out_path)


def plot_interactive_3d_decade_trajectory(billboard_df: pd.DataFrame, out_path: Path) -> None:
    """3D 年代风格轨迹图：x/y 为 PCA，z 为年代。"""
    feat_cols = [c for c in ["energy", "danceability", "happiness", "acousticness", "bpm", "loudness_d_b"] if c in billboard_df.columns]
    if "decade" not in billboard_df.columns or len(feat_cols) < 3:
        logging.warning("缺少 decade 或特征列不足，跳过 3D 年代轨迹图。")
        return

    temp = billboard_df.dropna(subset=["decade"] + feat_cols).copy()
    if temp.empty:
        return
    temp["decade"] = pd.to_numeric(temp["decade"], errors="coerce")
    temp = temp.dropna(subset=["decade"])
    if temp.empty:
        return
    temp["decade"] = temp["decade"].astype(int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(temp[feat_cols])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)
    temp["pc1"] = coords[:, 0]
    temp["pc2"] = coords[:, 1]

    # 控制交互大小
    temp = temp.sample(min(2500, len(temp)), random_state=42)
    temp["z_decade"] = temp["decade"].astype(float)
    temp["point_size"] = 4
    if "weeks_at_number_one" in temp.columns:
        w = pd.to_numeric(temp["weeks_at_number_one"], errors="coerce").fillna(1.0)
        w_min, w_max = float(w.min()), float(w.max())
        span = max(w_max - w_min, 1e-8)
        temp["point_size"] = 3.5 + (w - w_min) / span * 5.5

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=temp["pc1"],
            y=temp["pc2"],
            z=temp["z_decade"],
            mode="markers",
            marker=dict(
                size=temp["point_size"],
                color=temp["decade"],
                colorscale="Viridis",
                opacity=0.35,
                line=dict(width=0.2, color="rgba(255,255,255,0.4)")
            ),
            text=temp.get("song", temp.index.astype(str)),
            hovertemplate="年代: %{z}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            name="歌曲样本",
        )
    )

    centers = (
        temp.groupby("decade", as_index=False)
        .agg(pc1=("pc1", "mean"), pc2=("pc2", "mean"), n=("pc1", "size"))
        .sort_values("decade")
    )
    if len(centers) >= 2:
        n_min, n_max = float(centers["n"].min()), float(centers["n"].max())
        span = max(n_max - n_min, 1e-8)
        size_scaled = 10 + (centers["n"] - n_min) / span * 16
        fig.add_trace(
            go.Scatter3d(
                x=centers["pc1"],
                y=centers["pc2"],
                z=centers["decade"].astype(float),
                mode="lines+markers+text",
                marker=dict(size=size_scaled, color="#EF476F", line=dict(width=0.8, color="white")),
                line=dict(color="#EF476F", width=6),
                text=[f"{int(d)}年代" for d in centers["decade"]],
                textposition="top center",
                hovertemplate="年代: %{z}<br>质心PC1: %{x:.2f}<br>质心PC2: %{y:.2f}<extra></extra>",
                name="年代轨迹",
            )
        )

    evr = pca.explained_variance_ratio_
    fig.update_layout(
        template="plotly_white",
        font=PLOTLY_CHART_FONT,
        title="Billboard 冠军歌曲：3D 年代风格轨迹图",
        margin=dict(l=0, r=0, t=62, b=0),
        legend=dict(x=0.01, y=0.99),
        scene=dict(
            xaxis_title=f"主成分1（{evr[0]:.1%}）",
            yaxis_title=f"主成分2（{evr[1]:.1%}）",
            zaxis_title="年代",
            bgcolor="rgba(245,247,250,1)",
            camera=dict(eye=dict(x=1.6, y=1.4, z=1.2)),
        ),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logging.info("已保存交互图: %s", out_path)


def plot_interactive_3d_topic_cluster_bubble(
    billboard_df: pd.DataFrame, topics_long_df: pd.DataFrame, out_path: Path, top_n_topics: int = 18
) -> None:
    """3D 主题簇气泡图：基于主题共现聚类，将歌曲映射到主题簇。"""
    req_cols = [c for c in ["danceability", "energy", "happiness"] if c in billboard_df.columns]
    if len(req_cols) < 3 or topics_long_df.empty:
        logging.warning("缺少特征列或主题长表为空，跳过 3D 主题簇气泡图。")
        return

    topics_temp = topics_long_df.dropna(subset=["song_idx", "topic"]).copy()
    if topics_temp.empty:
        return

    top_topics = topics_temp["topic"].value_counts().head(top_n_topics).index.tolist()
    topics_temp = topics_temp[topics_temp["topic"].isin(top_topics)].copy()
    if topics_temp.empty or len(top_topics) < 4:
        return

    # 1) 主题级聚类
    by_song = topics_temp.groupby("song_idx")["topic"].apply(lambda s: sorted(set(s.dropna().tolist())))
    topic_to_i = {t: i for i, t in enumerate(top_topics)}
    mat = np.zeros((len(top_topics), len(top_topics)), dtype=np.int32)
    for t_list in by_song:
        for i in range(len(t_list)):
            for j in range(i + 1, len(t_list)):
                a, b = t_list[i], t_list[j]
                ia, ib = topic_to_i[a], topic_to_i[b]
                mat[ia, ib] += 1
                mat[ib, ia] += 1
    np.fill_diagonal(mat, 0)
    if int((mat > 0).sum()) == 0:
        logging.warning("主题共现矩阵为 0，跳过 3D 主题簇气泡图。")
        return

    affinity = mat.astype(float)
    affinity = affinity / max(float(affinity.max()), 1.0)
    k = max(3, min(6, len(top_topics) - 1))
    sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=42, assign_labels="kmeans")
    topic_cluster = sc.fit_predict(affinity)
    topic_cluster_map = {top_topics[i]: int(topic_cluster[i]) for i in range(len(top_topics))}

    # 2) 歌曲映射到主题簇（多数票）
    topics_temp["topic_cluster"] = topics_temp["topic"].map(topic_cluster_map)
    song_cluster = (
        topics_temp.groupby("song_idx")["topic_cluster"]
        .agg(lambda s: int(pd.Series(s).value_counts().index[0]))
        .rename("topic_cluster")
        .reset_index()
    )

    # 3) 合并歌曲特征
    bb = billboard_df.reset_index(drop=True).copy()
    bb["song_idx"] = bb.index
    merged = bb.merge(song_cluster, on="song_idx", how="inner")
    merged = merged.dropna(subset=req_cols)
    if merged.empty:
        return

    # 控制交互大小
    merged = merged.sample(min(1800, len(merged)), random_state=42)
    merged["cluster_label"] = merged["topic_cluster"].apply(lambda x: f"簇{x + 1}")

    color_pool = ["#FF6B6B", "#4ECDC4", "#FFD166", "#06D6A0", "#3A86FF", "#8338EC", "#F72585"]
    cluster_ids = sorted(merged["topic_cluster"].unique().tolist())
    cluster_color_map = {cid: color_pool[i % len(color_pool)] for i, cid in enumerate(cluster_ids)}
    merged["cluster_color"] = merged["topic_cluster"].map(cluster_color_map)

    fig = go.Figure()
    for cid in cluster_ids:
        sub = merged[merged["topic_cluster"] == cid]
        fig.add_trace(
            go.Scatter3d(
                x=sub["danceability"],
                y=sub["energy"],
                z=sub["happiness"],
                mode="markers",
                marker=dict(size=4.0, color=cluster_color_map[cid], opacity=0.28),
                name=f"簇{cid + 1}样本",
                hovertemplate="舞动性: %{x:.2f}<br>能量: %{y:.2f}<br>愉悦度: %{z:.2f}<extra></extra>",
            )
        )

    # 簇质心气泡
    cent = (
        merged.groupby("topic_cluster", as_index=False)
        .agg(
            danceability=("danceability", "mean"),
            energy=("energy", "mean"),
            happiness=("happiness", "mean"),
            n=("song_idx", "count"),
        )
        .sort_values("topic_cluster")
    )
    n_min, n_max = float(cent["n"].min()), float(cent["n"].max())
    span = max(n_max - n_min, 1e-8)
    bubble_size = 14 + (cent["n"] - n_min) / span * 26
    fig.add_trace(
        go.Scatter3d(
            x=cent["danceability"],
            y=cent["energy"],
            z=cent["happiness"],
            mode="markers+text",
            marker=dict(
                size=bubble_size,
                color=[cluster_color_map[int(c)] for c in cent["topic_cluster"]],
                opacity=0.95,
                line=dict(color="white", width=1.2),
            ),
            text=[f"簇{int(c)+1} (n={int(n)})" for c, n in zip(cent["topic_cluster"], cent["n"])],
            textposition="top center",
            name="簇中心",
            hovertemplate="簇中心<br>舞动性: %{x:.2f}<br>能量: %{y:.2f}<br>愉悦度: %{z:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        font=PLOTLY_CHART_FONT,
        title="Billboard 冠军歌曲：3D 主题簇气泡图",
        margin=dict(l=0, r=0, t=62, b=0),
        scene=dict(
            xaxis_title="舞动性",
            yaxis_title="能量",
            zaxis_title="愉悦度",
            bgcolor="rgba(245,247,250,1)",
            camera=dict(eye=dict(x=1.55, y=1.35, z=1.15)),
        ),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logging.info("已保存交互图: %s", out_path)


def plot_interactive_3d_with_album_panel(df: pd.DataFrame, out_path: Path) -> None:
    """Web 模块已移除：此函数保留为兼容占位。"""
    logging.info("Web/交互图模块已停用，跳过: %s", out_path)
    return


# =========================================================
# 11. 2D 专辑封面云图
# =========================================================
def plot_album_cover_cloud(df: pd.DataFrame, out_path: Path, top_n: int = 80) -> None:
    """Web/图片模块已移除：此函数保留为兼容占位。"""
    logging.info("专辑封面云图模块已停用，跳过: %s", out_path)
    return


# =========================================================
# 12. 报告输出
# =========================================================
def write_summary_report(
    spotify_df: pd.DataFrame,
    billboard_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    report_path: Path
) -> None:
    """生成简要文字报告，强调“关联不等于因果”。"""
    lines = []
    lines.append("# Music Hit Analysis Summary Report\n")
    lines.append(
        "\n> 注意：本报告仅描述数据中的关联、分布与共现模式，不构成因果结论。\n"
        "> 真实音乐传播过程还会受营销、平台曝光、发行窗口、受众结构等未观测变量影响。\n"
    )
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"- Spotify cleaned rows: {len(spotify_df):,}\n")
    lines.append(f"- Billboard cleaned rows: {len(billboard_df):,}\n")

    if "track_genre" in spotify_df.columns:
        top_genres = spotify_df["track_genre"].value_counts().head(10)
        lines.append("\n## 2. Top Spotify Genres\n")
        for genre, cnt in top_genres.items():
            lines.append(f"- {genre}: {cnt}\n")

    if "popularity" in spotify_df.columns:
        lines.append("\n## 3. Spotify Popularity\n")
        lines.append(f"- Mean popularity: {spotify_df['popularity'].mean():.2f}\n")
        lines.append(f"- Median popularity: {spotify_df['popularity'].median():.2f}\n")
        lines.append(f"- 90th percentile: {spotify_df['popularity'].quantile(0.90):.2f}\n")

    if "weeks_at_number_one" in billboard_df.columns:
        lines.append("\n## 4. Billboard #1 Duration\n")
        lines.append(f"- Mean weeks at #1: {billboard_df['weeks_at_number_one'].mean():.2f}\n")
        lines.append(f"- Max weeks at #1: {billboard_df['weeks_at_number_one'].max():.0f}\n")

    if not comparison_df.empty:
        lines.append("\n## 5. Spotify Hits vs Billboard #1 Songs\n")
        for _, row in comparison_df.iterrows():
            lines.append(
                f"- {row['feature']}: Spotify Hits={row['spotify_hit_mean']:.3f}, "
                f"Billboard #1={row['billboard_no1_mean']:.3f}\n"
            )
        lines.append(
            "\n> 对照表中 Billboard 的 danceability / energy / happiness（对应 valence）/ acousticness "
            "已与 Spotify 一致换算为 **0–1**；tempo 列为 **BPM**。\n"
        )
        lines.append("\nNote: Spotify `valence` and Billboard `happiness` are conceptually similar but not perfectly identical.\n")

    lines.append("\n## 6. Interpretation Boundary\n")
    lines.append("- 建议将图中结果作为“研究线索”而非“因果证明”。\n")
    lines.append("- 可在论文中报告“关联方向、效应大小、稳健性检验思路”，并明确潜在混杂变量。\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    logging.info(f"已保存报告: {report_path}")


# =========================================================
# 13. 主流程
# =========================================================
def main() -> None:
    dirs = build_output_dirs(OUTPUT_ROOT)
    setup_logging(dirs["logs"] / "music_hit_analysis_v3.log")

    logging.info("=" * 70)
    logging.info("Start music hit analysis pipeline")
    logging.info("=" * 70)

    # 尽早检测输出目录可写，避免跑完数据后才因权限失败（见《问题排查》1.3）。
    try:
        probe = dirs["root"] / ".music_analysis_write_probe"
        probe.write_text("", encoding="utf-8")
        if probe.exists():
            probe.unlink()
    except OSError as exc:
        logging.error("输出目录不可写，请检查 MUSIC_OUTPUT_ROOT 或磁盘权限: %s | %s", dirs["root"], exc)
        raise

    # -----------------------------
    # 步骤 1. 读取数据
    # -----------------------------
    spotify_raw = safe_read_csv(SPOTIFY_PATH)
    billboard_raw = safe_read_csv(BILLBOARD_PATH)
    topics_raw = safe_read_csv(TOPICS_PATH)

    # -----------------------------
    # 步骤 2. 清洗数据
    # -----------------------------
    spotify = clean_spotify(spotify_raw)
    billboard = clean_billboard(billboard_raw)
    topics = clean_topics(topics_raw)
    topics_long = extract_billboard_topics_long(billboard, topics)

    # 冒烟测试或低配机器：设置 MUSIC_SPOTIFY_MAX_ROWS 为正整数可随机下采样 Spotify 行数（见《问题排查》1.3）。
    _max_sp = os.environ.get("MUSIC_SPOTIFY_MAX_ROWS", "").strip()
    if _max_sp.isdigit():
        cap = int(_max_sp)
        if cap > 0 and len(spotify) > cap:
            spotify = spotify.sample(n=cap, random_state=42).copy()
            logging.info("MUSIC_SPOTIFY_MAX_ROWS=%s：Spotify 已下采样至 %s 行", _max_sp, len(spotify))

    # -----------------------------
    # 步骤 3. 保存清洗数据
    # -----------------------------
    save_dataframe(spotify, dirs["cleaned"] / "spotify_tracks_cleaned.csv")
    save_dataframe(billboard, dirs["cleaned"] / "billboard_cleaned.csv")
    save_dataframe(topics, dirs["cleaned"] / "topics_cleaned.csv")
    save_dataframe(topics_long, dirs["cleaned"] / "billboard_topics_exploded.csv")

    # -----------------------------
    # 步骤 4. 输出汇总表
    # -----------------------------
    spotify_genre_summary = (
        spotify.groupby("track_genre", as_index=False)
        .agg(song_count=("track_id", "count"), avg_popularity=("popularity", "mean"))
        .sort_values(["song_count", "avg_popularity"], ascending=[False, False])
    )
    save_dataframe(spotify_genre_summary, dirs["tables"] / "spotify_genre_summary.csv")

    billboard_decade_summary = (
        billboard.groupby("decade", as_index=False)
        .agg(
            songs=("song", "count"),
            avg_weeks_at_number_one=("weeks_at_number_one", "mean"),
            avg_danceability=("danceability", "mean"),
            avg_energy=("energy", "mean"),
            avg_happiness=("happiness", "mean") if "happiness" in billboard.columns else ("overall_rating", "mean")
        )
    )
    save_dataframe(billboard_decade_summary, dirs["tables"] / "billboard_decade_summary.csv")
    if not topics_long.empty:
        topic_summary = topics_long["topic"].value_counts().rename_axis("topic").reset_index(name="count")
        save_dataframe(topic_summary, dirs["tables"] / "billboard_topic_summary.csv")

    comparison = build_comparison_table(spotify, billboard)
    save_dataframe(comparison, dirs["tables"] / "spotify_vs_billboard_comparison.csv")

    # -----------------------------
    # 步骤 5. 静态图：Spotify
    # -----------------------------
    run_plot("spotify_popularity_distribution", lambda: plot_spotify_popularity_distribution(spotify, dirs["fig_spotify"] / "01_spotify_popularity_distribution.png"))
    run_plot("spotify_genre_counts", lambda: plot_spotify_genre_counts(spotify, dirs["fig_spotify"] / "02_spotify_top_genre_counts.png"))
    run_plot("spotify_genre_popularity", lambda: plot_spotify_genre_popularity(spotify, dirs["fig_spotify"] / "03_spotify_avg_popularity_by_genre.png"))
    run_plot("spotify_boxplot_by_hit_group", lambda: plot_spotify_boxplots_by_hit_group(spotify, dirs["fig_spotify"] / "04_spotify_boxplot_hit_group.png"))
    run_plot("spotify_violin_tier", lambda: plot_spotify_violin_by_tier(spotify, dirs["fig_spotify"] / "05_spotify_violin_popularity_tier.png"))
    run_plot("spotify_corr_heatmap", lambda: plot_spotify_corr_heatmap(spotify, dirs["fig_spotify"] / "06_spotify_corr_heatmap.png"))
    run_plot("spotify_duration_vs_popularity", lambda: plot_spotify_duration_vs_popularity(spotify, dirs["fig_spotify"] / "07_spotify_duration_vs_popularity.png"))
    run_plot("spotify_pair_density", lambda: plot_spotify_pair_density(spotify, dirs["fig_spotify"] / "08_spotify_density_danceability_energy.png"))
    run_plot("spotify_genre_radar", lambda: plot_spotify_genre_radar(spotify, dirs["fig_spotify"] / "09_spotify_genre_radar.png"))
    run_plot("spotify_stacked_tier_genre", lambda: plot_spotify_stacked_tier_by_genre(spotify, dirs["fig_spotify"] / "10_spotify_genre_tier_stacked.png"))
    run_plot("spotify_feature_map_pca", lambda: plot_spotify_feature_map_pca(spotify, dirs["fig_spotify"] / "11_spotify_feature_map_pca.png"))

    # -----------------------------
    # 步骤 6. 静态图：Billboard
    # -----------------------------
    run_plot("billboard_decade_trend", lambda: plot_billboard_decade_trend(billboard, dirs["fig_billboard"] / "01_billboard_decade_trends.png"))
    run_plot("billboard_weeks_distribution", lambda: plot_billboard_weeks_distribution(billboard, dirs["fig_billboard"] / "02_billboard_weeks_distribution.png"))
    run_plot("billboard_top_artists", lambda: plot_billboard_top_artists(billboard, dirs["fig_billboard"] / "03_billboard_top_artists.png"))
    run_plot("billboard_topic_distribution", lambda: plot_billboard_topic_distribution(billboard, dirs["fig_billboard"] / "04_billboard_lyrical_topics.png"))
    run_plot("billboard_bubble", lambda: plot_billboard_bubble(billboard, dirs["fig_billboard"] / "05_billboard_bubble_energy_danceability.png"))
    run_plot("billboard_corr_heatmap", lambda: plot_billboard_corr_heatmap(billboard, dirs["fig_billboard"] / "06_billboard_corr_heatmap.png"))
    run_plot("billboard_topic_decade_heatmap", lambda: plot_billboard_topic_decade_heatmap(topics_long, dirs["fig_billboard"] / "07_billboard_topic_decade_heatmap.png"))
    run_plot("billboard_topic_cooccurrence", lambda: plot_billboard_topic_cooccurrence(topics_long, dirs["fig_billboard"] / "08_billboard_topic_cooccurrence.png"))
    run_plot("billboard_topic_network", lambda: plot_billboard_topic_network(topics_long, dirs["fig_billboard"] / "09_billboard_topic_network.png"))
    run_plot("billboard_feature_ridgeline", lambda: plot_billboard_feature_ridgeline_by_decade(billboard, dirs["fig_billboard"] / "10_billboard_feature_ridgeline.png"))
    run_plot("billboard_topic_color_legend", lambda: plot_topic_color_legend(topics_long, dirs["fig_billboard"] / "11_billboard_topic_color_legend.png"))
    run_plot("billboard_decade_style_trajectory", lambda: plot_billboard_decade_style_trajectory(billboard, dirs["fig_billboard"] / "12_billboard_decade_style_trajectory.png"))
    run_plot("billboard_topic_community_graph", lambda: plot_billboard_topic_community_graph(topics_long, dirs["fig_billboard"] / "13_billboard_topic_community_graph.png"))

    # -----------------------------
    # 步骤 7. 对照图
    # -----------------------------
    run_plot("comparison_lollipop", lambda: plot_comparison_lollipop(comparison, dirs["fig_comparison"] / "01_spotify_vs_billboard_lollipop.png"))
    run_plot("comparison_gap_heatmap", lambda: plot_feature_gap_heatmap_by_decade(spotify, billboard, dirs["fig_comparison"] / "02_decade_feature_gap_heatmap.png"))
    run_plot("comparison_similarity_bar", lambda: plot_decade_similarity_to_spotify_hits(spotify, billboard, dirs["fig_comparison"] / "03_decade_similarity_to_spotify_hits.png"))
    run_plot("comparison_bootstrap_ci_hit_vs_low", lambda: plot_bootstrap_ci_hit_vs_low(spotify, dirs["fig_comparison"] / "04_spotify_hit_vs_low_bootstrap_ci.png"))
    run_plot("comparison_partial_corr_network", lambda: plot_billboard_partial_corr_network_by_decade(billboard, dirs["fig_comparison"] / "05_billboard_partial_corr_network_by_decade.png"))

    # -----------------------------
    # 步骤 8. 交互式 3D 图（HTML）
    # -----------------------------
    run_plot(
        "interactive_3d_spotify_cloud",
        lambda: plot_interactive_3d_cloud(spotify, dirs["fig_html"] / "01_spotify_3d_feature_cloud.html")
    )
    run_plot(
        "interactive_sankey_decade_topic",
        lambda: plot_billboard_decade_topic_sankey(topics_long, dirs["fig_html"] / "02_billboard_decade_topic_sankey.html")
    )
    run_plot(
        "interactive_3d_billboard_decade_trajectory",
        lambda: plot_interactive_3d_decade_trajectory(billboard, dirs["fig_html"] / "03_billboard_3d_decade_trajectory.html")
    )
    run_plot(
        "interactive_3d_billboard_topic_cluster_bubble",
        lambda: plot_interactive_3d_topic_cluster_bubble(
            billboard, topics_long, dirs["fig_html"] / "04_billboard_3d_topic_cluster_bubble.html"
        )
    )

    # -----------------------------
    # 步骤 9. 简要报告
    # -----------------------------
    write_summary_report(
        spotify_df=spotify,
        billboard_df=billboard,
        comparison_df=comparison,
        report_path=dirs["reports"] / "analysis_summary_report.md"
    )

    logging.info("全部完成。输出目录：%s", dirs["root"])


if __name__ == "__main__":
    main()







