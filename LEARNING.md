# Learning Guide

This document covers everything learned while building the Vehicle & Pedestrian Detector project. Written for anyone who wants to follow the same path.

---

## Who this is for

This project was built by a computer science undergraduate with intermediate Python experience and some exposure to machine learning tutorials. No prior computer vision experience was needed. If that sounds like you, this guide will be useful.

---

## How the project was approached

The project was built in phases — one working layer at a time. Each phase had a clear goal before moving to the next. This is the approach recommended for any ML project: get something running first, then improve it, then wrap it in an interface, then optimize.

The learning style was Socratic — understanding concepts by reasoning through them rather than memorizing definitions. For each new idea, the question was always "why does this work this way?" rather than "what does this do?"

---

## Phase 1 — Getting detection running

**What I did:** Ran a pretrained YOLOv8 model on a webcam and dashcam video.

**Key things learned:**

*What a pretrained model is.* YOLOv8 comes already trained on a dataset called COCO — 80 everyday object categories. It knows what a car looks like in general, but it has never seen a dashcam view, a traffic sign, or a pedestrian at night. That is what fine-tuning fixes.

*The closed-world assumption.* The model can only detect what it was trained to see. A fan in the background will never be detected — not because confidence is too low, but because "fan" simply does not exist in its world. This is why domain-specific training data matters.

*Confidence threshold.* Every detection comes with a confidence score between 0 and 1. Setting `conf=0.4` means "only show detections the model is at least 40% sure about." Lower = more detections, more false positives. Higher = fewer detections, more false negatives.

*False positives vs false negatives.* In safety-critical systems like autonomous driving, a false negative — missing a real pedestrian — is far more dangerous than a false positive — stopping for something that is not there. This tradeoff directly influences how you set the confidence threshold.

*OpenCV and Ultralytics.* Two libraries doing two different jobs. OpenCV handles video: opening files, reading frames, drawing boxes, showing windows. Ultralytics (YOLOv8) handles detection: understanding what is in each frame.

---

## Phase 2 — Training the model

**What I did:** Downloaded the BDD100K driving dataset and fine-tuned YOLOv8 on it for 50 epochs using a GPU.

**Key things learned:**

*What a dataset actually is.* BDD100K is not just images — it is images paired with answer keys. For every image there is a `.txt` annotation file containing one line per object:
class_id   x   y   width   height
These are the correct answers the model learns to produce.

*Why coordinates are normalized.* Bounding box coordinates are stored as fractions (0 to 1) rather than pixel values. A value of 0.5 always means the center of the image, regardless of resolution. This makes the model work across any image size.

*What neural network weights are.* YOLOv8n has about 3 million numbers called weights. Every prediction is just math: pixel values flow through layers of multiplications involving these weights and produce box predictions. Different weights produce different answers.

*How training works — the full loop:*
1. Show the model a batch of images
2. It predicts bounding boxes using its current weights
3. Compare predictions to the annotation answer keys — calculate the loss (the "wrongness score")
4. Backpropagation traces which weights contributed most to the error
5. Gradient descent nudges each weight slightly in the direction that reduces the error
6. Repeat for every batch, every epoch

*What an epoch is.* One complete pass through all training images. With 7,892 training images and batch size 16, one epoch is 494 weight updates.

*Why multiple epochs are needed.* One pass is not enough — the model sees each image briefly and the weight changes are small. Each subsequent pass refines the weights further, like re-reading study material to deepen understanding.

*Learning rate.* Controls how big each weight nudge is. Too high and the model overshoots — it bounces around the optimal weights without settling. Too low and training takes forever or gets stuck. Starting at 0.01 and decaying to a smaller value by the end (learning rate scheduling) is standard practice.

*Train vs validation split.* Training images are what the model learns from — weights change after each batch. Validation images are held out completely — the model never trains on them. After each epoch, the model is scored on validation images to measure genuine learning rather than memorization. This prevents overfitting.

*Class imbalance.* The training data had 7x more car instances than pedestrian instances. The model ended up much better at detecting cars (mAP50: 66.5%) than pedestrians (mAP50: 39.7%). The fix is more balanced training data.

*GPU vs CPU for training.* PyTorch training is fundamentally matrix multiplication — millions of independent calculations per batch. A CPU has 16 cores. An RTX 4070 has 4,608 CUDA cores. The same operation that takes minutes on CPU takes seconds on GPU.

---

## Phase 3 — Building the API and web app

**What I did:** Built a FastAPI backend serving the model and a Streamlit frontend that lets users upload images and see detections.

**Key things learned:**

*Why separate frontend and backend.* The detection logic lives in one place (the backend API). Any frontend — a web app today, a mobile app tomorrow — can use it without duplicating the model code. This is the standard architecture for any production ML system.

*What a REST API is.* A defined interface for communication between systems. The frontend sends an HTTP POST request with an image. The backend receives it, runs inference, and sends back a JSON response with detections. They never share code — only data.

*HTTP status codes.* 200 means success. 404 means not found. 422 means the request data was invalid. These are how servers and clients communicate about what happened.

*Why Base64 for images.* JSON can only carry text. A raw image is binary bytes. Base64 is a way of encoding binary data as text characters so it can travel inside a JSON response. The tradeoff: Base64 inflates file size by ~33%.

*Temporary files.* The YOLOv8 model expects a file path, not raw bytes. When an image arrives at the API it gets saved to a temporary file, the model reads it, then the file is deleted immediately. This prevents disk from filling up with every upload.

*H.264 codec for browser video.* OpenCV's default video codec (mp4v) works in desktop video players but browsers require H.264 (avc1). Discovered this when the annotated video played fine after download but showed a black screen in the browser.

---

## Phase 4 — Object tracking

**What I did:** Added DeepOcSort tracking so each detected object gets a persistent ID across frames.

**Key things learned:**

*Why detection alone is not enough.* A detector looks at each frame independently — it has no memory. Frame 1 sees car. Frame 2 sees car. It does not know it is the same car. Tracking adds that memory.

*How tracking works.* Three signals combined:
- Position — where was this object last frame?
- Size — how big was the bounding box?
- Appearance — what does the object look like? (handled by the ReID model)

*Kalman Filter.* Predicts where an object will be in the next frame based on its current velocity. When a car is briefly hidden behind another vehicle, the Kalman Filter keeps estimating its position so it can be matched when it reappears.

*ReID (Re-Identification).* A second neural network that creates a visual fingerprint for each tracked object. Used to match objects that disappear and reappear. Our ReID model (OSNet) was trained on person data — not vehicles — which is why tracking performance was limited.

*The occlusion problem.* When a car hides behind another for several seconds, even the best tracker struggles. Solving this properly requires a vehicle-trained ReID model and possibly a longer `max_age` parameter.

---

## Phase 5 — Video pipeline

**What I did:** Extended the web app to process full videos with frame-by-frame detection and return annotated video files.

**Key things learned:**

*Video is just images.* A 30fps video is 30 images per second. The model never sees "video" — it sees individual frames fed in sequence.

*VideoWriter.* OpenCV's tool for assembling annotated frames back into a video file. Must match the input video's fps, width, and height exactly or the output is distorted.

*UX for expensive operations.* Image detection is fast enough to run automatically on upload. Video processing takes minutes. Adding a confirmation button before video processing prevents accidentally kicking off a long job on the wrong file.

*FileResponse vs JSON.* Sending a large video as Base64 inside JSON inflates it by 33% and can crash browsers trying to hold it in memory. FastAPI's FileResponse streams the file directly — cleaner and faster for large binary files.

---

## Phase 6 — Optimization

**What I did:** Exported the model to ONNX format and measured the speedup.

**Key things learned:**

*What ONNX is.* Open Neural Network Exchange — a format for packaging trained model weights independently of the framework they were trained in. The same weights, repackaged for a leaner inference engine.

*Why ONNX is faster.* PyTorch carries training machinery even during inference: gradient tracking, weight update capability, dynamic graph construction. ONNX strips all of that away. ONNX Runtime runs a static, optimized computation graph — fewer operations, faster execution.

*Benchmarking properly.* Always run a warmup pass before timing — the first inference is always slower due to model loading and memory allocation. Then average over many runs (50 in our case) to get a stable measurement.

*The result.* 114ms per frame (PyTorch) → 44ms per frame (ONNX). 2.6x speedup on CPU. At 70mph, this is roughly 2 meters less blindness between perception updates.

---

## Version control with Git

**What I learned:**

Git is a timeline of your project. Every `commit` is a snapshot you can return to. Every `branch` is a parallel version for experimenting without breaking working code. Every `pull request` is a formal review before merging changes into the main branch.

Key habits formed during this project:
- Always create a new branch for each phase
- Write descriptive commit messages that explain what changed and why
- Use `.gitignore` to exclude large files (datasets, model weights, videos)
- Never commit credentials or API keys

The `.gitignore` patterns used:
data/

models/

outputs/

*.pt

*.onnx

*.mp4

pycache/

.env

---

## Libraries used

**PyTorch** — the math engine underneath everything. Handles matrix multiplication, automatic differentiation (backpropagation), and GPU acceleration via CUDA.

**Ultralytics (YOLOv8)** — the YOLOv8 implementation. Provides the model architecture, training loop, inference pipeline, and export tools. Our code configures and calls it; the actual neural network math lives here.

**OpenCV (cv2)** — computer vision utility library. Used for reading video files, extracting frames, drawing bounding boxes and text on frames, encoding video output, and Base64 image conversion.

**FastAPI** — modern Python web framework for building REST APIs. Handles HTTP routing, file uploads, request validation, and response formatting. Auto-generates interactive API documentation at `/docs`.

**Streamlit** — Python library for building web interfaces without writing HTML or JavaScript. Handles file uploaders, image display, video playback, tabs, and metrics cards.

**Roboflow** — platform for hosting and exporting computer vision datasets. Used to download BDD100K in YOLOv8 format at a manageable size.

**BoxMOT** — object tracking library. Provides DeepOcSort and other trackers. Used to add persistent IDs to detections across video frames.

**ONNX Runtime** — inference engine for ONNX models. Runs exported models faster than PyTorch by removing training overhead.

**CUDA** — NVIDIA's GPU computing platform. Enabled training on the RTX 4070, reducing training time from hours (CPU) to under 2 hours (GPU).

---

## Key metrics explained

**mAP50 (mean Average Precision at IoU 0.5)** — the standard object detection metric. Roughly: across all classes, how often does the model correctly find an object and draw a good box around it? Higher is better. Our model: 49.1% overall.

**Precision** — when the model says "this is a car," how often is it right? Our model: 68.5%.

**Recall** — of all real cars in the image, what fraction did the model find? Our model: 54.8%.

**IoU (Intersection over Union)** — measures how much a predicted box overlaps with the correct box. A value of 1.0 is a perfect match. Values above 0.5 are generally considered a correct detection.

**Loss** — the model's wrongness score during training. Three components: box loss (wrong position), cls loss (wrong class), dfl loss (wrong size). All three should decrease over epochs.

---

## What to build next

If you want to extend this project further:

- **Vehicle-trained ReID model** — swap OSNet for a model trained on the VeRi-776 vehicle re-identification dataset to fix tracking accuracy
- **TensorRT optimization** — NVIDIA's inference optimizer, even faster than ONNX on GPU
- **Real-time video stream** — replace file upload with a live RTSP stream input
- **Lane detection** — add semantic segmentation to detect drivable lanes alongside objects
- **Quantitative tracking evaluation** — measure tracking with HOTA and IDF1 metrics on the MOT17 benchmark