# 🍽️ FoodScan Challenge — Food Calorie Estimation from Images
### Deep Learning For Images — M2 IASD Apprenticeship | Université Paris Dauphine - PSL

---

## Overview

Given a single RGB image of a food dish, your goal is to build a deep learning model that predicts the total calorie content of the dish in kilocalories (kcal).

The dataset combines two real-world food image sources with verified calorie labels. Images vary in camera angle, lighting, dish complexity, and food type. The identity of the two sources will be disclosed after the competition deadline.

This project is evaluated across three components: your Kaggle leaderboard score, a hosted Streamlit application, and a live oral presentation.

---

## Competition

Join the Kaggle competition here:

🔗 **Link will be provided by email**

- All training must run inside a Kaggle notebook using Kaggle-provided GPU resources only
- Maximum 10 submissions per day
- Keep your notebook **private** during the competition — make it public only after the deadline
- No external datasets are permitted

---

## Dataset

The dataset is available directly inside the Kaggle competition.

| Split | Images | Calorie range | Mean calories |
|---|---|---|---|
| Train | 3,098 | 50 – 3,334 kcal | 338 kcal |
| Test | 547 | hidden | hidden |

**Files provided:**
- `train/images/` — RGB food dish images for training
- `train_labels.csv` — `image_id`, `filename`, `calories`
- `test/images/` — RGB food dish images for inference
- `test_ids.csv` — `image_id`, `filename` (no calorie labels)
- `sample_submission.csv` — required submission format

---

## Task

Predict the calorie content (kcal) for each image in `test/images/` and submit a CSV with the following format:

```
image_id, predicted_calories
test_0000, 450.0
test_0001, 320.5
```

The file must contain exactly **547 rows**.

---

## Metric

Submissions are scored on **Mean Absolute Error (MAE)**:

```
MAE = mean(|predicted_calories − true_calories|)
```

Lower is better. The leaderboard baseline (predicting the training mean for every image) scores **MAE = 308 kcal**. Your trained model must beat this significantly.

---

## Workflow

```
1. Train your model on Kaggle
        ↓
2. Export your model weights (.pt file) from Kaggle
        ↓
3. Build your Streamlit app locally
        ↓
4. Host your app on Streamlit Community Cloud (free)
        ↓
5. Send your deliverables by email before the deadline
```

---

## Streamlit Application

You must build a Streamlit app that:
- Accepts a food image upload
- Returns a calorie prediction in real time
- Is hosted and publicly accessible via a URL

A complete skeleton with TODOs is provided in the `streamlit_skeleton/` folder of this repository. Read `STREAMLIT.md` for full deployment instructions.

> **Note on model size:** If your model is large (e.g. ViT-B/16, Swin-T), hosting it directly may be slow or exceed memory limits on the free Streamlit tier. Consider reducing model size through quantization. If you encounter hosting issues, contact your instructor before the deadline.

---

## Deliverables

Submit **both of the following by email** to `mehyar.mlaweh@dauphine.eu` before:

### ⏰ Deadline: July 4, 2026 at 12:00 AM

| Deliverable | Details |
|---|---|
| Kaggle notebook URL | Make your notebook public on Kaggle before sending |
| Streamlit app URL | The hosted app must be live and accessible at submission time |

**Email subject line:**
```
[FoodScan] Team Submission — [Your Names]
```

---

## Presentation

### 📅 Date: July 6, 2026 (2:00 PM)


- **Format:** Online, slides required
- **Duration:** 5 minutes per team
- **Content:** Technical choices made, architecture justification, results analysis, what worked and what did not
- You are **not required** to send your slides in advance

---

## Grading

| Component | Weight | Details |
|---|---|---|
| 🏆 Kaggle leaderboard score | 60% | Based on private test set MAE — lower is better |
| 🖥️ Streamlit app | 20% | Live, hosted, functional at presentation time |
| 🎤 Presentation | 20% | 5 minutes, slides, technical depth and clarity |

---

## Important Dates

| Event | Date |
|---|---|
| Competition opens | Now |
| Submission deadline (notebook + app) | July 4, 2026 — 12:00 AM |
| Presentations  | July 6, 2026 (2:00 PM) |

---

## Contact

For any questions, technical issues, or problems with hosting your Streamlit app:

📧 **mehyar.mlaweh@dauphine.eu**

Do not wait until the last day to report a problem.

---

*Deep Learning For Images — M2 IASD Apprenticeship | Université Paris Dauphine - PSL*
