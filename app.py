import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# ──────────────────────────────────────────────────────────────
#  FOODSCAN — CALORIE ESTIMATOR
#  Complete every TODO section before deploying.
#  Your preprocessing must match EXACTLY what you used in training.
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FoodScan — Calorie Estimator",
    page_icon="🍽️",
    layout="centered"
)

# ── PAGE HEADER ───────────────────────────────────────────────
st.title("🍽️ FoodScan")
st.subheader("Food Calorie Estimator")
st.write(
    "Upload a photo of a food dish and the model will estimate "
    "its total calorie content."
)
st.divider()

# ── MODEL DEFINITION ─────────────────────────────────────────
# TODO: Paste your exact model class from your Kaggle notebook here.
# The architecture must match the weights you exported.
# Example structure below — replace with your actual model.

class CalorieEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define your backbone exactly as in your Kaggle notebook
        # Example:
        # self.backbone = models.efficientnet_b3(weights=None)
        # self.backbone.classifier = nn.Sequential(
        #     nn.Dropout(0.3),
        #     nn.Linear(self.backbone.classifier[1].in_features, 1)
        # )
        raise NotImplementedError("Replace this with your model definition")

    def forward(self, x):
        # TODO: Define your forward pass
        raise NotImplementedError("Replace this with your forward pass")


# ── MODEL LOADING ─────────────────────────────────────────────
# The model is loaded once and cached — do not remove @st.cache_resource

@st.cache_resource
def load_model():
    model = CalorieEstimator()

    # TODO: Choose ONE of the two loading options below

    # --- Option A: Model weights in the same folder as app.py ---
    # weights_path = "best_model.pt"
    # model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    # --- Option B: Model weights hosted on Hugging Face ----------
    # Use this if your model file exceeds 100 MB (GitHub file limit)
    
    model.eval()
    return model


# ── PREPROCESSING ─────────────────────────────────────────────
# TODO: Define your transform pipeline.
# This must be IDENTICAL to the validation transform you used during training.
# Common mistake: using different image size or normalization values here
# vs. what you used in training will degrade your predictions.

def get_transform():
    return transforms.Compose([
        # TODO: Replace 384 with your actual image size if different
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        # TODO: Use the same normalization as during training

    ])


# ── INFERENCE ────────────────────────────────────────────────
def predict(image: Image.Image, model: nn.Module) -> float:
    transform = get_transform()
    tensor    = transform(image).unsqueeze(0)  

    with torch.no_grad():
        output = model(tensor)

    predicted_calories = output.item()


    return round(predicted_calories, 1)


# ── MAIN UI ──────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a single food dish"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        st.write("")
        st.write("")
        with st.spinner("Estimating calories..."):
            try:
                model      = load_model()
                prediction = predict(image, model)

                st.metric(
                    label="Estimated Calories",
                    value=f"{prediction:.0f} kcal"
                )

            except NotImplementedError:
                st.error(
                    "Model not implemented yet. "
                    "Complete the TODO sections in app.py before deploying."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "FoodScan Challenge — Deep Learning For Images | "
    "M2 IASD Apprenticeship | Université Paris Dauphine - PSL"
)
