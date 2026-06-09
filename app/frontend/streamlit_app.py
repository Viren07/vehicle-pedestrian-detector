"""
Phase 3 — Streamlit frontend.
Run: streamlit run app/frontend/streamlit_app.py
"""

import streamlit as st, requests

st.set_page_config(page_title="Vehicle & Pedestrian Detector", layout="wide")
st.title("Vehicle & Pedestrian Detector")
st.caption("YOLOv8 · BDD100K fine-tuned")

uploaded = st.file_uploader("Upload a road image", type=["jpg","jpeg","png"])

if uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(uploaded, use_column_width=True)
    with col2:
        st.subheader("Detections")
        with st.spinner("Detecting..."):
            r = requests.post("http://localhost:8000/detect/image",
                              files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")})
        if r.status_code == 200:
            data = r.json()
            st.metric("Objects detected", data["count"])
            for d in data["detections"]:
                st.write(f"**{d['class']}** — {d['confidence']*100:.1f}%")
        else:
            st.error("Backend not running. Start with: uvicorn app.backend.main:app --reload")
