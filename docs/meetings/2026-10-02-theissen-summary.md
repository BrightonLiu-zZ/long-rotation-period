# 2026-10-02 与 Theissen 的谈话 — 要点

原始转写: `2026-10-02-theissen-transcript.txt` (说话人标签反了: "Instructor" = Brighton, "Student" = Theissen)。

## 他的决定

| 话题 | 他的结论 | 转写位置 |
|---|---|---|
| 输入数据 | **先用 SPOC light curve, 不用 image。** 先做容易的, 不够再做难的。他没有理由认为 SPOC 的 systematics 比自己提取的更多。 | 32:57, 36:00 |
| 留的口子 | 做完之后再判断 SPOC light curve 有没有问题、需不需要 FFI。 | 36:00 |
| 方法 | 模型预测 light curve 的下一个点 → 把曲线往后延长 → 延长后的曲线送进 periodogram, 长周期的 power 更明显。他叫它 "GAN", 泛指能预测的生成模型。 | 46:07–49:28 |
| Baseline | Lomb–Scargle periodogram 取峰, 对照 TARS 已发表的周期。 | 42:40 |
| 长周期真值 | **MEarth rotation period 样本** (M dwarf, 20–40 天以上)。只取周期, 不取图像。 | 43:53, 50:55 |
| 验证 | 只在 TESS 内部: 留出一部分星不训练; 把 sector 切两半, 用前半预测后半。跨 sector 更难 (sector 之间有 flux offset)。 | 09:59–12:24 |
| 样本范围 | 不限 M dwarf; 测周期与光谱型、年龄无关。 | 18:34–19:07 |
| 地/月位置作输入 | 不需要。只提醒: 13.7 天附近的信号要当 systematics 警惕。 | 20:46–22:02 |
| 其他望远镜 / LSST | 不要, 保持 TESS。 | 22:58 |
| 算力 | 先本地用 ~10 颗星跑通, 估出存储和计算量; 他可以把项目挂到他已有的超算额度上。 | 14:06–16:11 |
| 节奏 | 先要一个 quick result。 | 50:33 |

## 他提到的难点

- 周期超过观测窗口时, periodogram 的峰很弱, 不会是 "whopping signal" (42:01, 43:14)。
- 自己从 image 提取 light curve 要做很多主观选择: 哪些像素算星、背景怎么定、哪些 artifact 剔掉 (26:32–29:48)。

## 没解决 / 需要我自己核实

- 他说月光 "不会是周期信号", 当场也不确定 TESS 轨道周期。文献 (Hattori+2025 abstract) 说 13.7 天附近的背景信号正是长周期难测的原因。现在不争, 由 roadmap Step 3 用数据回答。
- 他不记得 TARS 用的是 SPOC light curve 还是 image (36:00)。按 09-10 的调研笔记, TARS 用的是 FFI (待我读原文方法部分确认)。
- "GAN" 的确切含义没对齐: 我两种都试 (复用 SWM 的预测模型 + 一个真 GAN)。
