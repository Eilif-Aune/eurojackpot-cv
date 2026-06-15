# Eurojackpot Draw Order Recovery and Statistical Analysis

I built a computer vision pipeline that recovers the order in which Eurojackpot lottery balls were drawn from the official YouTube draw videos, then ran every reasonable statistical test on the resulting dataset to see if anything in the draw process is predictable.

Spoiler: nothing is. But the journey of confirming that turned out to be more interesting than I expected.

## TL;DR

I trained two YOLOv8 models to detect lottery balls and read the digits printed on them, wrote a video pipeline that figures out the order balls arrive in the tube, and ran it on 14 years of Eurojackpot draws (2012 to 2026). The result is a dataset of 922 draws with their original draw order, which nobody else has.

Then I tested 13 separate hypotheses about whether the draws are predictable. None of them are.

One of the tests came back with p < 0.0001, which initially looked like a real discovery. It wasn't. It was a bug in my own pipeline. Figuring out why is the most interesting thing in this project.

## Background

Eurojackpot publishes results as sorted numbers, so you see "2, 17, 21, 25, 30" but you have no idea which one came out first, which came out second, and so on. The order is lost when the numbers are written down.

If you want to test certain hypotheses about lottery predictability, like whether consecutive draws influence each other or whether some positions favor certain numbers, you need that draw order. You cannot get it from any public dataset because nobody saves it.

But the draws are filmed and posted on YouTube, so the order is sitting there in the videos, just not in machine-readable form.

This project pulls the order out of the videos, builds a dataset from it, and uses that dataset to test every reasonable hypothesis I could think of.

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

## The CV Models

Two YOLOv8 models, both trained on frames I labeled by hand from the official Eurojackpot YouTube channel.

| Model | Training data | Test mAP@50 |
|-------|---------------|-------------|
| Ball detector | 78 frames | 0.987 |
| Digit detector | 612 digit instances | 0.984 |

## Draw Order Recovery

The trick is that balls arrive in the tube one at a time during the draw. If you can detect when each tube position first gets stably filled, you know the order they arrived in.

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

Each draw in the dataset has an `n_inferred` field that records how many of the five positions had to be guessed (when the digit detector couldn't confidently read the ball) versus directly detected.

| n_inferred | Draws | % |
|------------|-------|---|
| 0 (perfect) | 354 | 38.4% |
| 1 | 376 | 40.8% |
| 2 | 145 | 15.7% |
| 3 | 39 | 4.2% |
| 4 | 4 | 0.4% |
| 5 (total failure) | 3 | 0.3% |

79% of draws have at least 4 of 5 numbers directly detected. The `n_inferred` field looks like a bookkeeping detail. It turned out to be the most important field in the dataset, for reasons I'll get to.

## Statistical Analysis

I tested 13 separate hypotheses about whether the Eurojackpot draws are predictable. They cover a wide range of approaches: frequentist tests, bootstrap methods, Markov chains, hidden state models, neural networks, and frequency-domain analysis.

| # | Test | Method | Result |
|---|------|--------|--------|
| 1 | Marginal uniformity | Chi-square on 50 ball counts | p = 0.87 (uniform) |
| 2 | Conditional categorical | Bootstrap chi-square | p = 0.21 (no signal) |
| 3 | Markov chain z-scores | 50×50 transition matrix | Artifacts (Poisson asymmetry) |
| 4 | Markov predictive accuracy | Train/test, temporal split | At baseline (2.21% vs 1.87%) |
| 5 | Random Forest with rich features | 14 features, temporal split | Below baseline (1.71% vs 2.11%) |
| 6 | LSTM sequence model | 4-step input, predict 5th | Overfits, no generalization |
| 7 | Co-occurrence z-scores | All 1225 pairs | Within noise expectation |
| 8 | Co-occurrence predictive | Hit@K on held-out | Worse than uniform (-0.01pp) |
| 9 | Hidden Markov Model | K=2..7 with BIC + shuffle test | Gaussian Mixture artifact |
| 10 | Triplet/trigram patterns | Markov order 2 | Overfits, worse log-likelihood |
| 11 | **Position × Ball independence** | **Chi-square per position** | **p < 0.0001, but it's a pipeline artifact** |
| 12 | Per-position XGBoost + RF | 100+ features per position | All 5 positions below baseline |
| 13 | Spectral analysis | Welch's PSD + bootstrap | Fisher p = 0.33 (white noise) |

All thirteen came back negative. But one of them came back interestingly negative.

### The Pipeline Artifact (Test 11)

Test 11 nearly fooled me. The setup was simple: for each of the five draw positions, count how often each of the 50 balls appeared there, and run a chi-square test for independence. Under random draws, every ball should appear about equally often in every position.

The result was chi-square = 503, p < 0.0001. Position 5 in particular showed a striking pattern where high-numbered balls were massively over-represented in the final position. If I had stopped there and posted it, this would have looked like a real discovery about Eurojackpot.

It is not a real discovery. It is a bug in my own pipeline.

When the digit detector cannot confidently read a ball, the pipeline falls back to placing it at position 5. Two-digit balls (especially balls 40 and above) are harder to read than single-digit balls. The combination means: high-numbered balls fail digit detection more often, they get placed at position 5 by default, and that creates a fake correlation between ball value and final position.

The fix is to filter the dataset to draws where `n_inferred = 0`, meaning no ball needed to be guessed. That leaves 354 clean draws. Re-running the same chi-square test on this subset gives chi-square = 161, p = 0.97. All five positions individually have p > 0.7. The signal is gone.

This is why the `n_inferred` field matters. Without it, the original p < 0.0001 result would be impossible to distinguish from a real lottery bias. With it, the bug becomes detectable by comparing one slice of the data against another.

### Other Things Worth Mentioning

**The first chi-square false alarm (Test 2).** A standard parametric chi-square on prev → next transitions returned p = 0.039, which looked significant. The problem is that the four transitions within each draw are not independent observations, because they share a sampling-without-replacement constraint. A bootstrap that respects the within-draw structure gives p = 0.21. The parametric test was overstating significance by treating correlated data as independent.

**The Markov extreme cells (Test 3).** Several cells in the 50×50 transition matrix had z-scores above 3.0, which is more than expected by chance. They looked like signal but were really just Poisson asymmetry. With expected counts around 1.5 per cell, you can be much higher than expected but only slightly lower, since you can never have fewer than zero events. That asymmetry creates a heavy upper tail that mimics signal. A predictive model built on these "extreme" cells performed at random level on held-out data.

**The HMM Gaussian Mixture (Test 9).** An HMM with 7 hidden states had a much better BIC than a 1-state model, which initially suggested there might be hidden temporal structure in the draws. A shuffle test killed this. If I shuffle the chronological order of the draws before training the HMM, I get *better* test log-likelihood than on the real data. The HMM was approximating a multimodal feature distribution as a mixture of Gaussians. There was no actual temporal information being captured.

**XGBoost doing worse than random (Test 12).** I trained one XGBoost model per draw position, each with about 100 features (recency vectors, rolling frequencies, date features, previously drawn balls). All five models scored top-1 accuracy between 1.1% and 1.7%, while the uniform random baseline is 2.0% to 2.2%. So XGBoost was systematically beaten by chance. This is what overfitting on pure noise looks like: the model finds spurious patterns in the training data that actively hurt it on the test set. The top 10 feature importances were all between 0.011 and 0.015, basically uniform, which is what you would expect if the model has nothing real to learn.

**Spectral analysis (Test 13).** For each ball, I built a binary time series (1 if drawn at that draw, 0 if not) and ran Welch's power spectral density on it, then compared against a bootstrap null distribution. No ball had a peak above the 99% null threshold (0 out of 50, where you would expect 0.5 by chance). Fisher's combined test across all 50 balls gives p = 0.33. The aggregated spectrum is flat at the white-noise level expected for a Bernoulli(0.1) process. No periodicity, no seasonality, nothing.

## Predictive Model

The repo includes a deterministic predictive model that uses all the available signal sources I tested:

* Position-specific marginal distributions (not just sorted)
* Per-position Markov transitions
* Exponential time-weighting with a 365-day half-life
* Laplace smoothing
* Beam search over the full sequence

It's more sophisticated than just picking the most common balls. It also does not beat random on held-out data, which is the expected behavior given everything else in this project.

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
│   ├── 01_train_yolo.ipynb
│   ├── 02_statistical tests
├── data/
│   ├── eurojackpot_with_draw_order.parquet     # 923 draws × order + fasit
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

CV: YOLOv8 (ultralytics), OpenCV, imageio.

Stats and ML: numpy, pandas, scipy (signal, stats), scikit-learn, XGBoost, TensorFlow, hmmlearn.

Scraping: requests, regex, yt-dlp.

Infrastructure: Google Colab (T4 GPU), Google Drive.

## What I Learned

The most useful single thing this project taught me is to be paranoid about apparent signal. Five separate things in the analysis looked at first glance like real findings:

1. The parametric chi-square false alarm
2. The Markov extreme cells
3. The HMM hidden states
4. The position × ball signal
5. The XGBoost feature importances looking like they meant something

Every one of them had a clean explanation that did not involve the lottery being predictable. The position × ball case is the one that scared me most, because it had the lowest p-value and would have been the easiest to misinterpret. It also taught me the value of keeping audit fields in the dataset. The fix was not subtle statistics. It was just slicing the data on `n_inferred = 0` and re-running the same test on a clean subset.

## Limitations

* 923 draws may simply be too few to detect very weak signals if they exist
* The video resolution (480p) is the hard ceiling on detector accuracy
* The pipeline only handles main numbers (5 of 50), not the star numbers (2 of 12)
* Some early draws are missing from the fasit archive
* About 62% of position labels come from pipeline inference rather than direct detection. The `n_inferred` field lets anyone using the dataset filter to the clean 354-draw subset where this assumption is removed

## Wrapping Up

I spent a lot of work on a system designed to extract information that turned out to be unpredictable. That sounds like a failure, but it's actually the answer to the question I was asking. Across 13 different methods and roughly every analytical tool I know how to use, the Eurojackpot draws do not reveal any structure beyond uniform randomness.

The dataset itself is in this repo (`eurojackpot_with_draw_order.parquet`). If anyone reading this has a hypothesis they want to test, they have something nobody else does to test it on.

