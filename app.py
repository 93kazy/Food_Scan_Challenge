import os
import gc
import math

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ──────────────────────────────────────────────────────────────
#  FOODSCAN — CALORIE ESTIMATOR
#  2-model ConvNeXt ensemble: small@448 (5 folds) + base@384 (5 folds),
#  log1p target, hflip TTA, per-source affine calibration, 0.4/0.6 blend.
#  Reproduces the Kaggle pipeline (submission_ensemble, LB 33.43).
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="FoodScan — Calorie Estimator", page_icon="🍽️", layout="centered")

# ── ENSEMBLE CONFIG ───────────────────────────────────────────
# Two model groups, each a set of fold checkpoints averaged in log space, then
# per-source calibrated, then blended (0.4*small + 0.6*base) in kcal space.
#
# Weights are read from a local folder if present (local testing); otherwise each
# fold is downloaded once from a PUBLIC Hugging Face model repo (reliable, no quota).
#
# TODO (deployment): create a public Hugging Face model repo, upload the two model
# folders into it (small/ and base/ subfolders), and set HF_REPO below to
# "<your-hf-username>/<repo-name>".
#
# Per-source affine calibration (a, b) is fit on OOF with L1 loss — it undoes the
# weight-decay shrinkage. Source is inferred from image type: PNG == A (low-cal),
# JPG == B (high-cal). Values recomputed from oof_predictions_*.csv.
GROUPS = {
    "small": {
        "label": "ConvNeXt-Small (@448)",
        "weight": 0.4,
        "folds": [0, 1, 2, 3, 4],
        "local_dir": "Modèle small",
        "hf_prefix": "small",
        "calib": {"A": (1.053535, -5.2301), "B": (1.064414, -22.8851)},
    },
    "base": {
        "label": "ConvNeXt-V2-Base (@384)",
        "weight": 0.6,
        "folds": [0, 1, 2, 3, 4],
        "local_dir": "Modèle base",
        "hf_prefix": "base",
        "calib": {"A": (1.063964, -4.3660), "B": (1.071205, -18.6879)},
    },
}

# Public Hugging Face model repo holding the weights (subfolders small/ and base/).
# TODO (deployment): set this to your repo, e.g. "yourname/foodscan-weights".
HF_REPO = "Mazyn/foodscan"

MIN_CALORIES = 1.0            # floor for expm1 output (matches training)
N_MODELS = sum(len(g["folds"]) for g in GROUPS.values())


# ── PAGE HEADER ───────────────────────────────────────────────
st.title("🍽️ FoodScan")
st.subheader("Food Calorie Estimator")
st.write(
    "Upload a photo of a food dish and a 2-model ConvNeXt ensemble "
    "will estimate its total calorie content."
)
st.divider()


# ── MODEL DEFINITION ─────────────────────────────────────────
# Exact architecture from the Kaggle notebook: a timm backbone (global avg pool,
# no classifier) followed by a dropout + single linear regression head.
class CalorieRegressor(nn.Module):
    def __init__(self, backbone, drop_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False,
                                           num_classes=0, global_pool="avg")
        feat = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(feat, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


# ── PREPROCESSING ─────────────────────────────────────────────
# IDENTICAL to build_valid_transform in training: aspect-preserving resize so the
# longest side == image_size, center pad-to-square with black (value 0), then
# ImageNet normalization. image_size is read per-checkpoint (small=448, base=384).
@st.cache_resource(show_spinner=False)
def get_transform(image_size: int):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(),
        ToTensorV2(),
    ])


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    arr = np.array(image.convert("RGB"))          # RGB HWC uint8 (matches cv2 RGB)
    tensor = get_transform(image_size)(image=arr)["image"]
    return tensor.unsqueeze(0)                     # 1 x C x H x W


# ── WEIGHTS RESOLUTION (local folder or Hugging Face Hub) ─────
@st.cache_resource(show_spinner=False)
def resolve_weights() -> dict:
    """Return {group: {fold: path}}. Uses the local file if present (local
    testing), else downloads it once from the public HF repo. hf_hub_download
    caches and resumes, so it runs only on the first request."""
    from huggingface_hub import hf_hub_download
    out = {}
    for name, g in GROUPS.items():
        mapping = {}
        for fold in g["folds"]:
            local = os.path.join(g["local_dir"], f"fold{fold}_best.pth")
            if os.path.exists(local):
                mapping[fold] = local
                continue
            if HF_REPO.startswith("REPLACE"):
                raise RuntimeError(
                    "No local weights and HF_REPO is not set. "
                    "Set HF_REPO in app.py to your public Hugging Face repo."
                )
            mapping[fold] = hf_hub_download(
                repo_id=HF_REPO, filename=f"{g['hf_prefix']}/fold{fold}_best.pth"
            )
        out[name] = mapping
    return out


def load_model(path: str):
    """Build the regressor from the checkpoint's own metadata and load weights.
    Returns (model, image_size). The checkpoint stores backbone + image_size, so
    architecture and preprocessing size are never hardcoded."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CalorieRegressor(ckpt["backbone"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, int(ckpt["image_size"])


# ── INFERENCE ────────────────────────────────────────────────
def source_from_name(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    return "A" if ext == "png" else "B"   # png -> A, jpg/jpeg -> B


@torch.no_grad()
def predict_group_log(image: Image.Image, group: str, fold_paths: dict, progress=None) -> float:
    """Mean of the group's fold predictions in log1p space, hflip-TTA averaged.
    Models are loaded and freed one at a time to cap peak memory (~one model)."""
    logs = []
    for fold in GROUPS[group]["folds"]:
        model, size = load_model(fold_paths[fold])
        x = preprocess(image, size)
        out = model(x)
        out = (out + model(torch.flip(x, dims=[3]))) / 2      # hflip TTA in log space
        logs.append(out.item())
        del model, x
        gc.collect()
        if progress is not None:
            progress()
    return float(np.mean(logs))


def predict_calories(image: Image.Image, source: str, weights: dict, progress=None) -> dict:
    breakdown = {}
    total = 0.0
    for group, g in GROUPS.items():
        log_pred = predict_group_log(image, group, weights[group], progress)
        raw = max(MIN_CALORIES, math.expm1(log_pred))          # log1p -> kcal
        a, b = g["calib"][source]
        cal = max(MIN_CALORIES, a * raw + b)                   # per-source calibration
        breakdown[group] = cal
        total += g["weight"] * cal                             # 0.4/0.6 blend
    breakdown["total"] = total
    return breakdown


# ── MAIN UI ──────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a single food dish",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    detected = source_from_name(uploaded_file.name)

    source = detected   # calibration source, auto-detected from file type (PNG→A, JPG→B)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        st.write("")
        st.write("")
        bar = st.progress(0, text="Loading ensemble…")
        state = {"i": 0}

        def _tick():
            state["i"] += 1
            bar.progress(state["i"] / N_MODELS, text=f"Running model {state['i']}/{N_MODELS}…")

        try:
            weights = resolve_weights()
            result = predict_calories(image, source, weights, progress=_tick)
            bar.empty()
            st.metric(label="Estimated Calories", value=f"{result['total']:.0f} kcal")
            with st.expander("Ensemble breakdown"):
                for group, g in GROUPS.items():
                    st.write(f"{g['label']}: **{result[group]:.0f} kcal** × {g['weight']}")
                st.caption(f"Source **{source}** calibration applied.")
        except Exception as e:
            bar.empty()
            st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "FoodScan Challenge — Deep Learning For Images | "
    "M2 IASD Apprenticeship | Université Paris Dauphine - PSL"
)
