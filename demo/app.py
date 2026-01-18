import sys
import os

# ---------- Fix project import path ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------- Imports ----------
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

from physics_filters.orientation import dominant_orientation
from physics_filters.continuity import crack_length
from physics_filters.mask_utils import generate_binary_mask

# ---------- Page Config ----------
st.set_page_config(
    page_title="Physics-Aware Crack Detection (SHM Prototype)",
    layout="centered"
)

# ---------- Load CNN Model ----------
@st.cache_resource
def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load(
            "models/baseline_resnet/resnet18_sdnet_baseline.pth",
            map_location="cpu"
        )
    )
    model.eval()
    return model

model = load_model()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# ---------- UI Header ----------
st.title("Physics-Aware Crack Detection (SHM Prototype)")
st.write(
    "Inspection-support system that combines CNN-based crack detection "
    "with physics-aware reasoning to reduce false positives."
)

uploaded = st.file_uploader(
    "Upload a concrete surface image",
    type=["jpg", "png", "jpeg"]
)

# ---------- Main Pipeline ----------
if uploaded is not None:
    # ----- Load & show image -----
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image", width=600)

    # ----- CNN Prediction -----
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][1].item()  # crack confidence

    st.subheader("CNN Prediction")
    if pred == 1:
        st.write("🟥 Crack detected")
    else:
        st.write("🟩 No crack detected")

    # ----- Physics-Aware Analysis -----
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = generate_binary_mask(img_cv)

    angle = dominant_orientation(mask)
    length = crack_length(mask)

    st.subheader("Physics-Aware Analysis")
    st.write(f"Dominant orientation: {angle}")
    st.write(f"Estimated crack length: {length:.1f} px")

    # ----- Explainable Decision Logic -----
    st.subheader("Final Decision & Risk Assessment")

    reasons = []
    risk_level = "LOW"
    accepted = False

    VERY_LONG_THRESHOLD = 1500  # px safety override

    if angle is not None and length > 120:
        accepted = True
        reasons.append("Clear dominant crack orientation detected")
        reasons.append("Crack length above minimum safety threshold")

    elif angle is None and length > VERY_LONG_THRESHOLD:
        accepted = True
        reasons.append("Very long crack detected (safety override)")
        reasons.append("Irregular crack geometry accepted for safety")

    else:
        reasons.append("No reliable dominant orientation detected")
        reasons.append("Crack length below safety threshold")

    # ----- Risk Assignment -----
    if accepted and length > VERY_LONG_THRESHOLD:
        risk_level = "HIGH"
    elif accepted:
        risk_level = "MODERATE"
    elif length > 200:
        risk_level = "MONITOR"
    else:
        risk_level = "LOW"

    # ----- Display Results -----
    st.write(f"**CNN Crack Confidence:** {confidence:.2f}")

    if accepted:
        st.success("Decision: Crack CONFIRMED")
    else:
        st.warning("Decision: Rejected (likely false positive)")

    st.write(f"**Risk Level:** {risk_level}")

    st.markdown("**Decision Reasoning:**")
    for r in reasons:
        st.write(f"- {r}")

