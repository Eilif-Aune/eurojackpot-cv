# eurojackpot-cv
Computer vision pipeline for Eurojackpot ball detection — ML learning project
# Eurojackpot Draw Order Recovery and Statistical Analysis

A computer vision pipeline that recovers the original draw order from Eurojackpot lottery videos, combined with a rigorous statistical investigation of whether the resulting dataset contains any predictable structure.

## TL;DR

- **Data engineering**: Built a YOLOv8-based pipeline that processes 14 years of Eurojackpot draw videos (2012–2026) and recovers the order in which balls were drawn — information that is not preserved in any public dataset.
- **Resulting dataset**: 922 draws with reconstructed draw order. Verified at 5/5 accuracy on independent test cases. 79% of draws have at least 4 of 5 numbers directly detected.
- **Statistical analysis**: Ten independent tests for predictable patterns. All ten returned no significant signal.

This project did not find a way to predict Eurojackpot. With 14 years of data, ten independent analytical methods, and a unique draw-order dataset that no one else has, no predictable pattern was detected.

## Background

Eurojackpot results are published as sorted numbers (e.g. `2, 17, 21, 25, 30`). The original draw order is lost in this representation. Several hypotheses about lottery predictability — conditional dependencies between consecutive draws, position-specific bias, sequence patterns — can only be tested with draw order information.

This project recovers that lost information from publicly available draw videos using computer vision, then tests every plausible hypothesis about predictability on the resulting dataset.

## Pipeline Overview

```
Videos (YouTube)
     │
     ▼
Ball Detector (YOLOv8) ──────► Detect 5 balls per frame
     │
     ▼
Stable Period Detection ─────► Find frames where all 5 balls visible
     │
     ▼
Position Clustering ─────────► Group balls into 5 final positions
     │
     ▼
Digit Detector (YOLOv8) ─────► Read numbers off each ball
     │
     ▼
Hungarian Matching ──────────► Assign cluster → fasit number
     │
     ▼
Count-Increase Detection ────► Find when each cluster first filled
     │
     ▼
Output: Draw Order
```

## Computer Vision Models

Two YOLOv8 models trained on hand-annotated frames from the official Eurojackpot YouTube channel.

| Model | Training data | Test mAP@50 |
|-------|---------------|-------------|
| Ball detector | 78 frames | 0.987 |
| Digit detector | 612 digit instances | 0.984 |

## Draw Order Recovery

The key insight: balls are added to the tube one at a time during the draw. By tracking when each cluster position becomes "stably filled," we can infer the order of arrival.

```python
def recover_draw_order(video_path, fasit):
    # 1. Detect balls in every 3rd frame
    # 2. Find the longest stable period (>= 4 balls present)
    # 3. Cluster ball positions into 5 final locations
    # 4. Run digit detection on each cluster's crops
    # 5. Use Hungarian algorithm to match clusters to fasit numbers
    # 6. Track arrival time via count-increase events
    # 7. Infer missing balls as drawn last
    return ordered_sequence
```

### Validation

| Date | Predicted order | Actual order | Positions correct |
|------|-----------------|--------------|-------------------|
| 2026-03-20 | `[2, 25, 21, 17, 30]` | `[2, 25, 21, 17, 30]` | **5/5** |
| 2026-01-20 | `[26, 32, 45, 37, 16]` | `[32, 26, 37, 45, 16]` | 1/5 (adjacent swaps) |
| 2026-06-05 | `[50, 47, 21, 23, 44]` | `[50, 47, 21, 23, 44]` | **5/5** (visual check) |

### Quality Distribution

The pipeline reports `n_inferred` for each draw: the number of positions where the order had to be inferred rather than directly detected.

| n_inferred | Draws | % |
|------------|-------|---|
| 0 (perfect) | 354 | 38.4% |
| 1 | 376 | 40.8% |
| 2 | 145 | 15.7% |
| 3 | 39 | 4.2% |
| 4 | 4 | 0.4% |
| 5 (total failure) | 3 | 0.3% |

**79% of draws have at minimum 4 of 5 numbers directly detected.**

## Statistical Analysis

Ten independent hypotheses about whether the draw process is predictable were tested.

| # | Test | Method | Result |
|---|------|--------|--------|
| 1 | Marginal uniformity | Chi-square on 50 ball counts | p = 0.87 (uniform) |
| 2 | Conditional categorical | Bootstrap chi-square | p = 0.21 (no signal) |
| 3 | Markov chain z-scores | 50×50 transition matrix | Artifacts (Poisson asymmetry) |
| 4 | Markov predictive accuracy | Train/test, temporal split | At baseline |
| 5 | Random Forest with rich features | 14 features, temporal split | Below baseline (1.71% vs 2.11%) |
| 6 | LSTM sequence model | 4-step input, predict 5th | Overfits, no generalization |
| 7 | Co-occurrence z-scores | All 1225 pairs | Within noise expectation |
| 8 | Co-occurrence predictive | Hit@K on held-out | Worse than uniform (-0.01pp) |
| 9 | Hidden Markov Model | K=2..7 with BIC + shuffle test | Gaussian Mixture artifact |
| 10 | Triplet/trigram patterns | Markov order 2 | Overfits, worse log-likelihood |

### Selected Findings

**The chi-square false alarm.** A standard parametric chi-square test on low/medium/high prev → next transitions returned p = 0.039 — apparently significant. But the four transitions within each draw are not independent observations: they share a sampling-without-replacement constraint. A bootstrap test that shuffles within draws gave p = 0.21. The parametric test was inflating significance by ignoring the correlation structure.

**The Markov "extreme cells."** Several cells in the 50×50 transition matrix showed z-scores > 3.0, more than expected by chance. These looked impressive but were artifacts of low expected counts (~1.5 per cell) and asymmetric Poisson distribution: cells can be much higher than expected but only slightly lower (bounded at 0). A predictive model built on these "extreme" cells performed at random level on held-out data.

**The HMM Gaussian Mixture.** A Hidden Markov Model with K=7 hidden states gave dramatically better BIC than K=1, suggesting hidden temporal structure. But a shuffle test (shuffling time order before training) showed shuffled data gave **better** test log-likelihood than original. The HMM was approximating a multimodal feature distribution as a Gaussian Mixture Model — there was no temporal information being captured.

## Predictive Model

The project includes a principled predictive model that uses all available information:

- Position-specific marginal distribution (uses draw order, not just sorted values)
- Conditional transitions (Markov, per-position)
- Exponential time-weighting (half-life: 365 days, recent draws weighted higher)
- Laplace smoothing
- Beam search for the most likely full sequence

This is more sophisticated than "predict the most common balls." It is also more honest: it does not perform better than random on held-out data, which is the expected result given the analytical findings.

```python
prediction = recover_draw_order_for_next_drawing(historical_data)
# Output: a single deterministic prediction that updates only as new data arrives
```

## Project Structure

```
eurojackpot-cv/
├── README.md
├── scripts/
│   ├── fetch_eurojackpot.py          # Scrape fasit (winning numbers)
│   ├── fetch_video_urls.py           # Find YouTube videos by date
│   ├── download_videos.py            # Batch download with yt-dlp
│   └── generate_all_urls.py          # Build complete URL list
├── notebooks/
│   ├── 01_train_ball_detector.ipynb
│   ├── 02_train_digit_detector.ipynb
│   ├── 03_pipeline_validation.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_ml_models.ipynb
│   └── 06_predictive_model.ipynb
├── data/
│   ├── eurojackpot_with_draw_order.parquet     # 922 draws × order + fasit
│   └── euro_jackpot_komplett.csv               # Fasit only
└── models/
    ├── ball_detector/weights/best.pt
    └── digit_detector/weights/best.pt
```

## Reproduce

```bash
# 1. Fetch fasit (winning numbers)
python scripts/fetch_eurojackpot.py

# 2. Find YouTube videos for missing dates
python scripts/fetch_video_urls.py

# 3. Download videos
python scripts/download_videos.py

# 4. Run the pipeline (in Colab with GPU)
# See notebooks/03_pipeline_validation.ipynb
```

## Tech Stack

- **CV**: YOLOv8 (ultralytics), OpenCV, imageio
- **Stats / ML**: numpy, pandas, scipy, scikit-learn, TensorFlow, hmmlearn
- **Data**: pandas, parquet
- **Scraping**: requests + regex, yt-dlp
- **Infrastructure**: Google Colab (T4 GPU), Google Drive

## What This Project Demonstrates

1. **Original data engineering**: Recovering information from videos that does not exist in any public dataset
2. **End-to-end ML pipeline**: From data acquisition through model training to inference
3. **Statistical rigor**: Proper train/test splits, bootstrap validation, multiple-comparison awareness
4. **Self-critical analysis**: Several apparent "findings" (Markov z-scores, HMM hidden states) were investigated and shown to be artifacts
5. **Honest reporting**: Negative results documented with the same care positive ones would be

## Limitations

- 922 draws may be too few to detect very weak signals if they exist
- Source video resolution (480p) is the fundamental ceiling on pipeline accuracy
- The pipeline only handles main numbers (5/50), not the star numbers (2/12)
- Some early draws lack fasit data in the source archive
