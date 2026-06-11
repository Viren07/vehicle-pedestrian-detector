"""
Phase 5 — Streamlit frontend with image and video detection.
Run: streamlit run app/frontend/streamlit_app.py
"""
import base64
import streamlit as st, requests

st.set_page_config(page_title="Vehicle & Pedestrian Detector", page_icon="🚗", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* Subtle grid background */
.stApp {
    background-color: #0A0A0F;
    background-image:
        linear-gradient(rgba(124, 77, 255, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 77, 255, 0.05) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* Hero header — glassmorphism + glow */
.header-box {
    position: relative;
    background: rgba(20, 18, 35, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(124, 77, 255, 0.25);
    border-radius: 20px;
    padding: 2.4rem 2.2rem;
    margin-bottom: 1.8rem;
    overflow: hidden;
}
.header-box::before {
    content: "";
    position: absolute;
    top: -60%; left: -10%;
    width: 60%; height: 200%;
    background: radial-gradient(circle, rgba(124,77,255,0.25) 0%, transparent 70%);
    animation: drift 10s ease-in-out infinite alternate;
}
@keyframes drift { from {transform: translateX(0);} to {transform: translateX(60%);} }
.header-box h1 {
    color: #FFFFFF; margin: 0; font-size: 2.4rem; font-weight: 700;
    letter-spacing: -0.02em;
}
.header-box h1 span {
    background: linear-gradient(90deg, #7C4DFF, #00E5FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-box p {
    color: #8B8B9E; margin: 0.5rem 0 0 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
}
.badge {
    display: inline-block;
    background: rgba(124, 77, 255, 0.15);
    border: 1px solid rgba(124, 77, 255, 0.4);
    color: #B39DFF;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    margin-top: 0.9rem;
    margin-right: 0.4rem;
}

/* Stat cards — glass with glow hover */
.stat-card {
    background: rgba(20, 18, 35, 0.65);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(124, 77, 255, 0.2);
    border-radius: 16px;
    padding: 1.3rem 1rem;
    text-align: center;
    transition: all 0.25s ease;
}
.stat-card:hover {
    transform: translateY(-5px);
    border-color: rgba(124, 77, 255, 0.7);
    box-shadow: 0 8px 32px rgba(124, 77, 255, 0.25);
}
.stat-card h3 {
    margin: 0; font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #7C4DFF, #00E5FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-card p {
    color: #8B8B9E; margin: 0.3rem 0 0 0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}

/* Detection rows */
.det-item {
    background: rgba(20, 18, 35, 0.65);
    border: 1px solid rgba(124, 77, 255, 0.15);
    border-left: 3px solid #7C4DFF;
    padding: 0.55rem 1rem;
    margin: 0.35rem 0;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    transition: all 0.18s ease;
}
.det-item:hover {
    border-left-color: #00E5FF;
    transform: translateX(4px);
    background: rgba(124, 77, 255, 0.08);
}

/* Tabs styling */
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] { color: #7C4DFF !important; }

/* Buttons */
.stButton button, .stDownloadButton button {
    background: linear-gradient(90deg, #7C4DFF, #651FFF);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 1.4rem; font-weight: 500;
    transition: all 0.2s ease;
}
.stButton button:hover, .stDownloadButton button:hover {
    box-shadow: 0 6px 24px rgba(124, 77, 255, 0.45);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="header-box">
    <h1>Vehicle & Pedestrian <span>Detector</span></h1>
    <p>> YOLOv8 · fine-tuned on BDD100K · autonomous driving perception</p>
    <div>
        <span class="badge">PyTorch</span>
        <span class="badge">FastAPI</span>
        <span class="badge">CUDA</span>
        <span class="badge">mAP50 49.1%</span>
    </div>
</div>
""", unsafe_allow_html=True)

API = "http://localhost:8000"

tab_image, tab_video, tab_about = st.tabs(["🖼️ Image Detection", "🎬 Video Detection", "📘 About"])

# ---------- IMAGE TAB ----------
with tab_image:
    uploaded = st.file_uploader("Upload a road image", type=["jpg","jpeg","png"], key="image_upload")

    if uploaded:
        with st.spinner("Detecting..."):
            r = requests.post(f"{API}/detect/image",
                              files={"file": (uploaded.name, uploaded.getvalue(), "image/jpeg")})
        if r.status_code == 200:
            data = r.json()

            # Stat cards row
            classes = [d["class"] for d in data["detections"]]
            c1, c2, c3, c4 = st.columns(4)
            for col, (label, value) in zip(
                [c1, c2, c3, c4],
                [("Total Objects", data["count"]),
                 ("Cars", classes.count("car")),
                 ("Pedestrians", classes.count("pedestrian")),
                 ("Signs & Lights", classes.count("traffic sign") + classes.count("traffic light"))]):
                col.markdown(f'<div class="stat-card"><h3>{value}</h3><p>{label}</p></div>',
                             unsafe_allow_html=True)

            st.write("")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(uploaded, use_container_width=True)
            with col2:
                st.subheader("Annotated")
                st.image(base64.b64decode(data["annotated_image"]), use_container_width=True)

            with st.expander("Detection details"):
                for d in data["detections"]:
                    st.markdown(
                        f'<div class="det-item"><b>{d["class"]}</b> — {d["confidence"]*100:.1f}% confidence</div>',
                        unsafe_allow_html=True)
        else:
            st.error("Backend not running. Start with: uvicorn app.backend.main:app --reload")

# ---------- VIDEO TAB ----------
with tab_video:
    uploaded_video = st.file_uploader("Upload a road video", type=["mp4","avi","mov"], key="video_upload")

    if uploaded_video:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.video(uploaded_video)

        if st.button("🚀 Run Detection on Video"):
            with st.spinner("Processing video — this may take a few minutes..."):
                r = requests.post(f"{API}/detect/video",
                                  files={"file": (uploaded_video.name, uploaded_video.getvalue(), "video/mp4")})
            if r.status_code == 200:
                with col2:
                    st.subheader("Annotated")
                    st.video(r.content)
                st.download_button("⬇️ Download Annotated Video", r.content,
                                   file_name="annotated_video.mp4", mime="video/mp4")
            else:
                st.error("Video processing failed. Is the backend running?")

# ---------- ABOUT TAB ----------
# ---------- ABOUT TAB ----------
with tab_about:
    st.markdown("""
    ## What is this project?

    This is a computer vision system that can **look at road footage and identify what's in it** —
    the same fundamental technology that self-driving cars from companies like Waymo and Tesla
    use to understand the world around them.

    Upload a photo or video taken from a car's dashboard, and the system will find and label every
    **car, pedestrian, traffic light, and traffic sign** it can see — drawing boxes around each one
    with a confidence score showing how sure it is.

    ---

    ## How does it work?

    **1. The Brain — YOLOv8**

    At the core is a neural network called YOLO ("You Only Look Once"). It looks at an entire
    image in a single pass and predicts where objects are and what they are. This makes it fast
    enough to process video.

    **2. The Training — BDD100K Dataset**

    A neural network is only as good as what it has learned from. This model was trained on
    nearly 10,000 real driving images from the Berkeley DeepDrive dataset — including daytime,
    nighttime, and bad weather scenes — so it learns what roads actually look like in the real world.

    **3. The Engine Room — FastAPI Backend**

    The trained model runs behind a REST API. When you upload an image or video here, it gets sent
    to the backend, the model analyzes it frame by frame, and the results come back to your browser.

    **4. The Interface — This Website**

    Built with Streamlit. It talks to the backend, shows you the results, and lets you download
    annotated videos.

    ---

    ## How was it built?

    This project was built in phases, each one adding a layer:

    | Phase | What was built |
    |---|---|
    | 1 | Ran a pretrained detector on webcam and dashcam video |
    | 2 | Fine-tuned the model on 9,900 road images using an RTX 4070 GPU |
    | 3 | Built the API backend and this web interface |
    | 4 | Added object tracking with persistent IDs (experimental) |
    | 5 | Added full video upload, processing, and download |

    ---

    ## How well does it perform?

    The standard metric is **mAP@50** — roughly, "how often does the model correctly find an object."

    | Class | mAP@50 | In plain English |
    |---|---|---|
    | Car | 66.5% | Best — it saw 7x more cars than anything else during training |
    | Traffic Sign | 48.1% | Decent — signs are small but distinctive |
    | Traffic Light | 42.2% | Harder — small and often far away |
    | Pedestrian | 39.7% | Weakest — fewest training examples |

    ---

    ## Honest limitations

    - **Pedestrian detection is the weakest** because the training data had far fewer pedestrians than cars. More balanced data would fix this.
    - **Object tracking loses IDs during long occlusions** — when a car is hidden behind another for several seconds. The re-identification model was trained on people, not vehicles.
    - **Not real-time yet** — processing runs at roughly 10–15 fps. Real autonomous vehicles need 30+ fps, which would require model optimization (ONNX/TensorRT).

    ---

    ## Tech Stack

    `Python` · `PyTorch` · `YOLOv8` · `OpenCV` · `FastAPI` · `Streamlit` · `CUDA` · `Roboflow` · `BoxMOT` · `Git`

    ## Source Code

    The full code, training scripts, and documentation:
    [github.com/Viren07/vehicle-pedestrian-detector](https://github.com/Viren07/vehicle-pedestrian-detector)
    """)