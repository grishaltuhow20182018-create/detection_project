from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
FAST_MODEL_PATH = ROOT / "fast_model.pt"
THINKING_MODEL_PATH = ROOT / "thinking_model.pt"


@st.cache_resource
def load_models() -> tuple[YOLO, YOLO]:
    return YOLO(FAST_MODEL_PATH), YOLO(THINKING_MODEL_PATH)


def choose_model(request_text: str, image: Image.Image, force_model: str) -> tuple[YOLO, str, str]:
    fast_model, thinking_model = load_models()
    if force_model == "Fast model":
        return fast_model, "fast", "You forced the fast model for a quick result."
    if force_model == "Thinking model":
        return thinking_model, "thinking", "You forced the thinking model for higher accuracy."

    normalized = request_text.lower().strip()
    blurry_keywords = {
        "blurry",
        "blur",
        "low quality",
        "hard",
        "difficult",
        "small objects",
        "tiny objects",
        "accurate",
        "quality",
        "maximum",
        "best",
        "smart",
        "complex",
        "размытая",
        "точная",
        "максимум",
        "лучше",
    }
    fast_keywords = {
        "fast",
        "quick",
        "speed",
        "rapid",
        "simple",
        "clear",
        "easy",
        "быстро",
        "простой",
        "четкая",
    }

    if any(keyword in normalized for keyword in blurry_keywords):
        return thinking_model, "thinking", "The request suggests a harder or lower-quality image, so I picked the more accurate model."

    if any(keyword in normalized for keyword in fast_keywords):
        return fast_model, "fast", "The request asks for speed or a simple image, so I picked the faster model."

    width, height = image.size
    if min(width, height) < 900:
        return thinking_model, "thinking", "The image is relatively small, so I picked the more accurate model to preserve detail."

    return fast_model, "fast", "No strong quality warning was given, so I picked the faster model for a quick result."


def analyze_image(image: Image.Image, request_text: str, confidence: float, force_model: str):
    model, model_name, reason = choose_model(request_text, image, force_model)
    result = model.predict(source=image, conf=confidence, verbose=False)[0]

    detections = []
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append(
                {
                    "label": names.get(class_id, str(class_id)),
                    "confidence": round(float(box.conf.item()), 4),
                    "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )

    annotated_image = Image.fromarray(result.plot()[:, :, ::-1])
    return model_name, reason, detections, annotated_image


st.set_page_config(page_title="Dual Model Detection", page_icon="V", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, rgba(55, 215, 178, 0.18), transparent 28%),
                        radial-gradient(circle at 85% 10%, rgba(125, 168, 255, 0.18), transparent 26%),
                        linear-gradient(135deg, #06101b 0%, #0b1424 42%, #060b14 100%);
        }
        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 24px;
            border: 1px solid rgba(145, 170, 220, 0.18);
            background: rgba(11, 18, 32, 0.82);
        }
        .small-note {
            color: #9ba8c7;
            font-size: 0.95rem;
            line-height: 1.7;
        }
        .badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(55, 215, 178, 0.14);
            color: #37d7b2;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="badge">Dual-model detection</div>
      <h1 style="margin:0 0 0.6rem 0; font-size: clamp(2.3rem, 5vw, 4.3rem); line-height: 0.95; letter-spacing: -0.05em;">Fast when it is easy. Accurate when it is not.</h1>
      <p class="small-note" style="margin:0; max-width: 72ch;">
        Upload an image, describe what you need, and the app will choose between the fast model and the thinking model.
        Streamlit Cloud can run this directly because the whole app lives in one Streamlit entry file.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

left, right = st.columns([0.36, 0.64], gap="large")

with left:
    request_text = st.text_area(
        "Your request",
        placeholder="Example: use maximum quality, image is blurry, detect everything",
        height=140,
    )
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    confidence = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    force_model = st.selectbox("Model preference", ["Auto", "Fast model", "Thinking model"])

    analyze_clicked = st.button("Analyze image", use_container_width=True, type="primary")

with right:
    if uploaded_file is None:
        st.info("Upload an image to see detections here.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if analyze_clicked:
            with st.spinner("Running model inference..."):
                model_name, reason, detections, annotated_image = analyze_image(
                    image=image,
                    request_text=request_text,
                    confidence=confidence,
                    force_model=force_model,
                )

            st.success(f"Model used: {model_name}")
            st.caption(reason)

            a, b, c = st.columns(3)
            a.metric("Detections", len(detections))
            b.metric("Confidence", f"{confidence:.2f}")
            c.metric("Image size", f"{image.width} x {image.height}")

            st.image(annotated_image, caption="Annotated result", use_container_width=True)

            if detections:
                st.subheader("Detected objects")
                st.dataframe(detections, use_container_width=True, hide_index=True)
            else:
                st.warning("No objects detected at the current confidence threshold.")
        else:
            st.warning("Set your options and click Analyze image to run the models.")

with st.expander("How model selection works", expanded=False):
    st.markdown(
        """
        - If you force a model in the sidebar, that model is always used.
        - If the request mentions blur, difficulty, accuracy, or quality, the thinking model is chosen.
        - If the request mentions speed, clarity, or simplicity, the fast model is chosen.
        - Otherwise smaller images default to the thinking model and larger clear images default to the fast model.
        """
    )