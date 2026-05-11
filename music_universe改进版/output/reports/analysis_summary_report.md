# Music Hit Analysis Summary Report

> 注意：本报告仅描述数据中的关联、分布与共现模式，不构成因果结论。
> 真实音乐传播过程还会受营销、平台曝光、发行窗口、受众结构等未观测变量影响。
## 1. Dataset Overview
- Spotify cleaned rows: 89,741
- Billboard cleaned rows: 1,177

## 2. Top Spotify Genres
- acoustic: 1000
- afrobeat: 999
- alt-rock: 999
- ambient: 999
- cantopop: 999
- tango: 999
- chicago-house: 998
- forro: 998
- bluegrass: 998
- study: 998

## 3. Spotify Popularity
- Mean popularity: 33.20
- Median popularity: 33.00
- 90th percentile: 60.00

## 4. Billboard #1 Duration
- Mean weeks at #1: 2.94
- Max weeks at #1: 19

## 5. Spotify Hits vs Billboard #1 Songs
- danceability: Spotify Hits=0.595, Billboard #1=62.947
- energy: Spotify Hits=0.638, Billboard #1=60.477
- valence: Spotify Hits=0.484, Billboard #1=61.776
- acousticness: Spotify Hits=0.279, Billboard #1=30.019
- tempo: Spotify Hits=120.433, Billboard #1=115.856

Note: Spotify `valence` and Billboard `happiness` are conceptually similar but not perfectly identical.

## 6. Interpretation Boundary
- 建议将图中结果作为“研究线索”而非“因果证明”。
- 可在论文中报告“关联方向、效应大小、稳健性检验思路”，并明确潜在混杂变量。
