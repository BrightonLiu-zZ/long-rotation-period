# Roadmap — long rotation period

写于 2026-10-03。时间不固定, 所以按**步骤**排, 不按周排。每步标了大概工时和 "做到这就可以停" 的产出。

**两条原则**
1. 花最少的时间, 把 Theissen 要的东西大致做好: SPOC light curve → 预测延长 → periodogram。
2. 用一个便宜的检验 (Step 3) 回答他自己留的问题: "SPOC light curve 有没有问题, 需不需要回到像素"。结果两种都可能, 不预设。

---

## 步骤

| # | 做什么 | 产出 | 工时 |
|---|---|---|---|
| 1 | **真值表。** MEarth 周期 (Newton+2016 北天 387 颗; Newton+2018 南天) 对到 TIC; 数出多少颗有 SPOC 2-min light curve、各几个 sector、周期分布。再从 TARS 取一小批短周期星做对照。 | 一张表 + 几个计数 | 2–3 h |
| 2 | **Baseline。** 先 ~10 颗星, 再全表: 对 PDCSAP 做 Lomb–Scargle, 取峰, 对照真值。按 P < 13 d / 13–27 d / > 27 d 分三档报 "10% 以内的比例"。 | 一张结果表 + 几张 periodogram 图 = **给 Theissen 的 quick result** | 2–3 h |
| 3 | **"SPOC 够不够" 检验。** 同一批星, 把 PDCSAP 换成 SPOC 文件里的 SAP_FLUX 再跑一遍 Step 2。看长周期信号在哪一步消失。 | 一张 SAP vs PDCSAP 对照表 → 走下面的分支 | 2–3 h |
| 4 | **模型 A: 复用 SWM 的预测模型。** 沿用 encoder + GRU 的代码和训练循环; 数据改成按 30 分钟 bin 的整 sector 曲线。按他的方案验证: 留出星; 前半 sector 预测后半。 | 预测误差 vs "直接复制上一段"; 延长后 LS vs 不延长 LS | 4–8 h |
| 5 | **模型 B: 一个小 GAN。** 同样的数据、同样的验证。只在 A 跑通后做。 | 同上, 多一行 | 4–8 h |
| 6 | **汇报 + 决定。** 一张总表给 Theissen; 根据 Step 3 的结果提 (或不提) 像素这一步。 | Slack 消息 + 下次会议 | 1–2 h |

**最小可交差版本 = Step 1–3** (约 6–9 小时)。到这里已经有 baseline、有他要的 quick result、也有 "要不要像素" 的证据。

---

## Step 3 的三种结果

| 结果 | 含义 | 下一步 |
|---|---|---|
| SAP 里有长周期信号, PDCSAP 里没了 | SPOC 的去系统误差步骤把信号一起抹掉了 | 有证据回到更早的数据。提议用 **SPOC 的 Target Pixel File** (像素小视频, 仍是 SPOC 产品, aperture 和背景 SPOC 已定好) → 回到 CV |
| 两者都有 | SPOC light curve 够用 | 项目保持最小, 做完 Step 4–6 收尾; CV 经验另找 |
| 两者都没有 | 要么信息本身不在, 要么 systematics 盖住了 | 看 13.7 天附近的 power; 像素是自然的下一步, 但证据弱于第一种 |

---

## 要诚实对待的两点

- **"延长再做 periodogram" 不产生新的观测。** 延长段是模型按学到的规律续写的。它若有用, 只能是因为模型学到了 "这类信号通常怎么延续"。所以 Step 4/5 必须带两行对照: (a) 不延长直接 LS; (b) 用一条简单正弦拟合去延长。模型赢不了 (b), 就没有可报告的增益。
- **前半 sector 预测后半, 测的是预测能力, 不是长周期。** 长周期只能靠 MEarth 真值来测。两个都报, 不混在一起。

---

## 还没核实的

- MEarth 星里有多少有 SPOC 2-min 数据 → Step 1 自己数。
- "PDCSAP 会抹掉 sector 尺度的信号" 这个说法在多篇文章里出现, 但我还没定位到确切出处 → Step 3 自己测, 不引用。
- Newton+2018 南天样本的星数和周期范围 → Step 1 读原文。

## 参考

- MEarth 北天周期, Newton+2016: https://arxiv.org/abs/1511.00957 (387 颗, 0.1–140 天)
- MEarth 南天周期, Newton+2018: https://arxiv.org/abs/1807.09365
- TARS: https://arxiv.org/abs/2603.05586
- Hattori+2025 (13.7 天限制, FFI + ZTF 真值): https://arxiv.org/abs/2505.10376
- 会前的调研 (像素方向、相关工作): `docs/prep/`
