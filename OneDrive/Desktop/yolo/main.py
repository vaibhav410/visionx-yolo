import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import gradio as gr
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
import base64
from datetime import datetime
import time

# ==========================================
#           CONFIGURATION
# ==========================================
MODEL_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\yolo\best.pt"

# ==========================================
#           LOAD YOLO MODEL
# ==========================================
def load_model():
    """Load YOLO model with error handling"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        return None
    try:
        print(f"🔄 Loading model from: {MODEL_PATH}")
        yolo_model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return yolo_model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ==========================================
#      OPENCV PREPROCESSING FUNCTIONS
# ==========================================
def apply_opencv_processing(img, enhance_contrast, apply_blur, edge_detection, brightness, denoise, sharpen):
    """Apply OpenCV image preprocessing operations"""
    processed = img.copy()
    
    if brightness != 1.0:
        processed = cv2.convertScaleAbs(processed, alpha=brightness, beta=0)
    
    if denoise:
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    
    if enhance_contrast:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        processed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    if apply_blur:
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
    
    if sharpen:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)
    
    if edge_detection:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return processed

# ==========================================
#        YOLO INFERENCE FUNCTION
# ==========================================
def run_yolo_detection(img_bgr, conf_threshold=0.25, iou_threshold=0.45):
    """Run YOLO detection on image"""
    if model is None:
        return [], None, {}
    
    results = model(img_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
    result = results[0]
    
    detections = []
    class_counts = {}
    
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = model.names[cls_id]
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        detections.append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })
    
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    return detections, annotated_rgb, class_counts

# ==========================================
#          FASTAPI APPLICATION
# ==========================================
app = FastAPI(
    title="VisionX Safety API",
    description="Professional Object Detection API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    """Fully Responsive API Home page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VisionX Safety API</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                width: 100%;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                border-radius: 30px;
                padding: 60px;
                box-shadow: 0 20px 80px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .logo {
                width: 100px;
                height: 100px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 25px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 48px;
                margin: 0 auto 30px;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            }
            h1 {
                text-align: center;
                color: #ffffff;
                font-size: 48px;
                font-weight: 900;
                margin-bottom: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .subtitle {
                text-align: center;
                color: rgba(255, 255, 255, 0.8);
                font-size: 20px;
                margin-bottom: 40px;
            }
            .status-badge {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                background: rgba(34, 197, 94, 0.2);
                border: 2px solid rgba(34, 197, 94, 0.5);
                color: #22c55e;
                padding: 12px 24px;
                border-radius: 50px;
                font-weight: 600;
                margin: 0 auto 40px;
                width: fit-content;
                font-size: 16px;
            }
            .status-dot {
                width: 12px;
                height: 12px;
                background: #22c55e;
                border-radius: 50%;
                animation: blink 1.5s ease-in-out infinite;
            }
            @keyframes blink {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .stat-item {
                text-align: center;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 25px;
            }
            .stat-value {
                font-size: 36px;
                font-weight: 900;
                color: #667eea;
                margin-bottom: 5px;
            }
            .stat-label {
                color: rgba(255, 255, 255, 0.7);
                font-size: 14px;
                text-transform: uppercase;
            }
            .endpoints {
                background: rgba(0, 0, 0, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px;
                margin-top: 40px;
            }
            .endpoints h2 {
                color: #ffffff;
                font-size: 24px;
                margin-bottom: 20px;
                text-align: center;
            }
            .endpoint-item {
                display: flex;
                align-items: center;
                gap: 15px;
                padding: 15px;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                margin-bottom: 12px;
                flex-wrap: wrap;
            }
            .endpoint-method {
                padding: 6px 12px;
                border-radius: 8px;
                font-weight: 700;
                font-size: 12px;
                min-width: 60px;
                text-align: center;
                flex-shrink: 0;
            }
            .endpoint-method.get { background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); color: white; }
            .endpoint-method.post { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; }
            .endpoint-path {
                font-family: 'Courier New', monospace;
                color: #a78bfa;
                font-weight: 600;
                flex: 1;
                min-width: 100px;
            }
            .endpoint-desc {
                color: rgba(255, 255, 255, 0.6);
                font-size: 13px;
            }
            .footer {
                text-align: center;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.6);
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .container { 
                    padding: 30px 20px; 
                    border-radius: 20px;
                }
                .logo {
                    width: 80px;
                    height: 80px;
                    font-size: 40px;
                }
                h1 { 
                    font-size: 32px; 
                }
                .subtitle {
                    font-size: 16px;
                }
                .stats {
                    grid-template-columns: 1fr;
                    gap: 15px;
                }
                .stat-value {
                    font-size: 28px;
                }
                .endpoint-item {
                    flex-direction: column;
                    align-items: flex-start;
                }
                .endpoint-path {
                    width: 100%;
                }
            }
            
            @media (max-width: 480px) {
                .container {
                    padding: 20px 15px;
                }
                h1 {
                    font-size: 28px;
                }
                .status-badge {
                    font-size: 14px;
                    padding: 10px 20px;
                }
                .endpoints h2 {
                    font-size: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🛰️</div>
            <h1>VisionX Safety API</h1>
            <p class="subtitle">AI-Powered Object Detection System</p>
            <div class="status-badge">
                <span class="status-dot"></span>
                System Online
            </div>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value">YOLOv8</div>
                    <div class="stat-label">AI Model</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">99.9%</div>
                    <div class="stat-label">Uptime</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">< 100ms</div>
                    <div class="stat-label">Response</div>
                </div>
            </div>
            <div class="endpoints">
                <h2>📡 API Endpoints</h2>
                <div class="endpoint-item">
                    <span class="endpoint-method post">POST</span>
                    <span class="endpoint-path">/detect</span>
                    <span class="endpoint-desc">Object Detection</span>
                </div>
                <div class="endpoint-item">
                    <span class="endpoint-method get">GET</span>
                    <span class="endpoint-path">/health</span>
                    <span class="endpoint-desc">Health Check</span>
                </div>
                <div class="endpoint-item">
                    <span class="endpoint-method get">GET</span>
                    <span class="endpoint-path">/docs</span>
                    <span class="endpoint-desc">API Documentation</span>
                </div>
            </div>
            <div class="footer">
                <p><strong>VisionX Safety v2.0</strong></p>
                <p style="margin-top: 10px;">Professional Object Detection System</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
def health_check():
    """Check API and model health"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "api_version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "model_classes": len(model.names) if model else 0
    }

@app.post("/detect")
async def detect_api(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    preprocessing: bool = Form(False),
    brightness: float = Form(1.0),
    enhance_contrast: bool = Form(False),
    denoise: bool = Form(False),
    blur: bool = Form(False),
    sharpen: bool = Form(False),
    edge_detection: bool = Form(False)
):
    """Detect objects in uploaded image"""
    if model is None:
        return JSONResponse(
            {"error": "Model not loaded", "status": "failed"},
            status_code=500
        )
    
    try:
        start_time = time.time()
        
        img_bytes = await file.read()
        img_pil = Image.open(BytesIO(img_bytes))
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        if preprocessing:
            img_bgr = apply_opencv_processing(
                img_bgr, enhance_contrast, blur, edge_detection, brightness, denoise, sharpen
            )
        
        detections, annotated_rgb, class_counts = run_yolo_detection(img_bgr, confidence, iou_threshold)
        
        avg_confidence = (
            sum(d["confidence"] for d in detections) / len(detections)
            if detections else 0
        )
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "status": "success",
            "total_detections": len(detections),
            "average_confidence": round(avg_confidence, 3),
            "class_counts": class_counts,
            "detections": detections,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "status": "failed"},
            status_code=500
        )

# ==========================================
#         GRADIO DETECTION FUNCTION
# ==========================================
def detect_gradio(image, enable_preprocessing, enhance_contrast, apply_blur, 
                  edge_detection, brightness, denoise, sharpen, confidence_threshold, iou_threshold):
    """Gradio detection function"""
    
    if image is None:
        return None, None, create_error_html("⚠️ Please upload an image first.")
    
    if model is None:
        return None, None, create_error_html(f"⚠️ Model not loaded! Path: {MODEL_PATH}")
    
    try:
        start_time = time.time()
        
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if enable_preprocessing:
            processed_img = apply_opencv_processing(
                img_bgr, enhance_contrast, apply_blur, edge_detection, brightness, denoise, sharpen
            )
        else:
            processed_img = img_bgr.copy()
        
        preprocessed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        preprocessed_pil = Image.fromarray(preprocessed_rgb)
        
        detections, annotated_rgb, class_counts = run_yolo_detection(processed_img, confidence_threshold, iou_threshold)
        output_pil = Image.fromarray(annotated_rgb)
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        html = create_detection_html(detections, class_counts, enable_preprocessing, 
                                     brightness, enhance_contrast, denoise, apply_blur, 
                                     sharpen, edge_detection, processing_time)
        
        return preprocessed_pil, output_pil, html
        
    except Exception as e:
        return None, None, create_error_html(f"⚠️ Error: {str(e)}")

def create_detection_html(detections, class_counts, preprocessing, brightness, 
                         contrast, denoise, blur, sharpen, edges, proc_time):
    """Create fully responsive HTML for detection results"""
    if not detections:
        return """
        <div class="no-detection-container">
            <div style='font-size: 64px; margin-bottom: 20px;'>🔍</div>
            <div style='color: rgba(255,255,255,0.9); font-size: 20px; font-weight: 600; margin-bottom: 10px;'>
                No Objects Detected
            </div>
            <div style='color: rgba(255,255,255,0.6); font-size: 14px;'>
                Try adjusting the confidence threshold or preprocessing settings
            </div>
        </div>
        
        <style>
            .no-detection-container {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 2px solid rgba(239, 68, 68, 0.3);
                border-radius: 20px;
                padding: 40px 20px;
                text-align: center;
            }
            
            @media (max-width: 768px) {
                .no-detection-container {
                    padding: 30px 15px;
                }
                .no-detection-container div:first-child {
                    font-size: 48px !important;
                }
                .no-detection-container div:nth-child(2) {
                    font-size: 18px !important;
                }
            }
        </style>
        """
    
    avg_conf = sum(d["confidence"] for d in detections) / len(detections) * 100
    
    html = f"""
    <div class="detection-results-container">
        
        <!-- Stats Grid - Fully Responsive -->
        <div class="stats-grid">
            <div class="stat-box stat-objects">
                <div class="stat-value">{len(detections)}</div>
                <div class="stat-label">Objects</div>
            </div>
            
            <div class="stat-box stat-confidence">
                <div class="stat-value">{avg_conf:.1f}%</div>
                <div class="stat-label">Confidence</div>
            </div>
            
            <div class="stat-box stat-classes">
                <div class="stat-value">{len(class_counts)}</div>
                <div class="stat-label">Classes</div>
            </div>
            
            <div class="stat-box stat-time">
                <div class="stat-value">{proc_time:.0f}ms</div>
                <div class="stat-label">Time</div>
            </div>
        </div>
        
        <!-- Class Distribution -->
        <div class="section-container">
            <div class="section-title">📊 Class Distribution</div>
            <div class="class-distribution">
    """
    
    max_count = max(class_counts.values())
    colors = ['#22c55e', '#3b82f6', '#a855f7', '#f59e0b', '#ef4444', '#06b6d4']
    
    for idx, (cls_name, count) in enumerate(class_counts.items()):
        percentage = (count / max_count) * 100
        color = colors[idx % len(colors)]
        html += f"""
            <div class="class-item">
                <div class="class-header">
                    <span class="class-name">{cls_name}</span>
                    <span class="class-count" style="color: {color};">{count}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="background: {color}; width: {percentage}%;"></div>
                </div>
            </div>
        """
    
    html += "</div></div>"
    
    # Preprocessing info
    if preprocessing:
        applied = []
        if brightness != 1.0: applied.append(f"💡 Brightness ({brightness:.1f}x)")
        if contrast: applied.append("🎨 CLAHE")
        if denoise: applied.append("🧹 Denoise")
        if blur: applied.append("🌫️ Blur")
        if sharpen: applied.append("✨ Sharpen")
        if edges: applied.append("🔲 Edges")
        
        if applied:
            html += """
            <div class="section-container preprocessing-section">
                <div class="section-title">🔧 Preprocessing Applied</div>
                <div class="preprocessing-tags">
            """
            
            for item in applied:
                html += f'<span class="preprocessing-tag">{item}</span>'
            
            html += "</div></div>"
    
    # Detection details
    html += """
        <div class="section-container">
            <div class="section-title">🎯 Detection Details</div>
            <div class="detection-list">
    """
    
    for det in detections:
        conf_percent = det["confidence"] * 100
        if conf_percent >= 80:
            color = "#22c55e"
            icon = "✅"
        elif conf_percent >= 50:
            color = "#f59e0b"
            icon = "⚠️"
        else:
            color = "#ef4444"
            icon = "❌"
        
        html += f"""
            <div class="detection-item" style="border-left-color: {color};">
                <div class="detection-header">
                    <div class="detection-name">
                        <span class="detection-icon">{icon}</span>
                        <span class="detection-class">{det["class"]}</span>
                    </div>
                    <div class="detection-confidence" style="background: {color}22; border-color: {color}44; color: {color};">
                        {conf_percent:.1f}%
                    </div>
                </div>
                <div class="detection-box">
                    📦 Box: [{det["box"][0]}, {det["box"][1]}, {det["box"][2]}, {det["box"][3]}]
                </div>
            </div>
        """
    
    html += "</div></div></div>"
    
    # Add comprehensive responsive styles
    html += """
    <style>
        /* Main Container */
        .detection-results-container {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid rgba(102, 126, 234, 0.3);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .stat-box {
            border-radius: 12px;
            padding: 12px;
            text-align: center;
            min-width: 0;
        }
        
        .stat-objects { background: rgba(34, 197, 94, 0.1); border: 2px solid rgba(34, 197, 94, 0.3); }
        .stat-confidence { background: rgba(59, 130, 246, 0.1); border: 2px solid rgba(59, 130, 246, 0.3); }
        .stat-classes { background: rgba(168, 85, 247, 0.1); border: 2px solid rgba(168, 85, 247, 0.3); }
        .stat-time { background: rgba(249, 115, 22, 0.1); border: 2px solid rgba(249, 115, 22, 0.3); }
        
        .stat-value {
            font-size: 24px;
            font-weight: 900;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .stat-objects .stat-value { color: #22c55e; }
        .stat-confidence .stat-value { color: #3b82f6; }
        .stat-classes .stat-value { color: #a855f7; }
        .stat-time .stat-value { color: #f97316; }
        
        .stat-label {
            color: rgba(255,255,255,0.7);
            font-size: 11px;
            text-transform: uppercase;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* Section Container */
        .section-container {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }
        
        .section-title {
            color: #667eea;
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 12px;
        }
        
        /* Class Distribution */
        .class-distribution {
            display: grid;
            gap: 8px;
            max-height: 200px;
            overflow-y: auto;
            padding-right: 8px;
        }
        
        .class-item {
            background: rgba(255,255,255,0.02);
            border-radius: 8px;
            padding: 10px;
        }
        
        .class-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            align-items: center;
            gap: 10px;
        }
        
        .class-name {
            color: #fff;
            font-weight: 600;
            font-size: 13px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex: 1;
        }
        
        .class-count {
            font-weight: 700;
            font-size: 13px;
            white-space: nowrap;
            flex-shrink: 0;
        }
        
        .progress-bar {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            height: 6px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 8px;
            transition: width 0.3s ease;
        }
        
        /* Preprocessing Section */
        .preprocessing-section {
            background: rgba(102, 126, 234, 0.1);
            border: 2px solid rgba(102, 126, 234, 0.3);
        }
        
        .preprocessing-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .preprocessing-tag {
            background: rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(102, 126, 234, 0.4);
            color: #a78bfa;
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 11px;
            font-weight: 600;
            white-space: nowrap;
        }
        
        /* Detection List */
        .detection-list {
            max-height: 350px;
            overflow-y: auto;
            overflow-x: hidden;
            padding-right: 8px;
        }
        
        .detection-item {
            background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.1);
            border-left: 3px solid;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
        }
        
        .detection-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .detection-name {
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
            min-width: 0;
        }
        
        .detection-icon {
            font-size: 18px;
            flex-shrink: 0;
        }
        
        .detection-class {
            color: #fff;
            font-weight: 700;
            font-size: 14px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .detection-confidence {
            padding: 4px 12px;
            border-radius: 16px;
            font-weight: 700;
            font-size: 12px;
            white-space: nowrap;
            flex-shrink: 0;
            border: 2px solid;
        }
        
        .detection-box {
            color: rgba(255,255,255,0.5);
            font-size: 11px;
            font-family: monospace;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* Custom Scrollbar */
        .class-distribution::-webkit-scrollbar,
        .detection-list::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        .class-distribution::-webkit-scrollbar-track,
        .detection-list::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        }
        
        .class-distribution::-webkit-scrollbar-thumb,
        .detection-list::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.5);
            border-radius: 4px;
        }
        
        .class-distribution::-webkit-scrollbar-thumb:hover,
        .detection-list::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.7);
        }
        
        /* ============================================ */
        /*           RESPONSIVE MEDIA QUERIES           */
        /* ============================================ */
        
        /* Tablets (768px and below) */
        @media (max-width: 768px) {
            .detection-results-container {
                padding: 16px;
                border-radius: 16px;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
            }
            
            .stat-value {
                font-size: 20px;
            }
            
            .stat-label {
                font-size: 10px;
            }
            
            .section-title {
                font-size: 14px;
            }
            
            .class-name,
            .class-count {
                font-size: 12px;
            }
            
            .detection-class {
                font-size: 13px;
            }
            
            .detection-confidence {
                font-size: 11px;
                padding: 3px 10px;
            }
            
            .preprocessing-tag {
                font-size: 10px;
                padding: 3px 8px;
            }
        }
        
        /* Mobile (480px and below) */
        @media (max-width: 480px) {
            .detection-results-container {
                padding: 12px;
                border-radius: 12px;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
                gap: 8px;
            }
            
            .stat-box {
                padding: 10px;
            }
            
            .stat-value {
                font-size: 22px;
            }
            
            .stat-label {
                font-size: 10px;
            }
            
            .section-container {
                padding: 12px;
            }
            
            .section-title {
                font-size: 13px;
                margin-bottom: 10px;
            }
            
            .class-distribution {
                max-height: 180px;
            }
            
            .class-item {
                padding: 8px;
            }
            
            .class-name {
                font-size: 11px;
            }
            
            .class-count {
                font-size: 12px;
            }
            
            .detection-list {
                max-height: 300px;
            }
            
            .detection-item {
                padding: 10px;
            }
            
            .detection-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
            }
            
            .detection-name {
                width: 100%;
            }
            
            .detection-class {
                font-size: 12px;
            }
            
            .detection-icon {
                font-size: 16px;
            }
            
            .detection-confidence {
                font-size: 10px;
                padding: 3px 8px;
            }
            
            .detection-box {
                font-size: 10px;
            }
            
            .preprocessing-tags {
                gap: 4px;
            }
            
            .preprocessing-tag {
                font-size: 9px;
                padding: 3px 6px;
            }
        }
        
        /* Extra small devices (360px and below) */
        @media (max-width: 360px) {
            .stat-value {
                font-size: 18px;
            }
            
            .detection-class {
                font-size: 11px;
            }
            
            .class-name {
                font-size: 10px;
            }
        }
    </style>
    """
    
    return html

def create_error_html(message):
    """Create responsive error HTML"""
    return f"""
    <div class="error-container">
        <div class="error-icon">⚠️</div>
        <div class="error-message">{message}</div>
    </div>
    
    <style>
        .error-container {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid rgba(239, 68, 68, 0.5);
            border-radius: 20px;
            padding: 30px 20px;
            text-align: center;
        }}
        
        .error-icon {{
            font-size: 48px;
            margin-bottom: 15px;
        }}
        
        .error-message {{
            color: #ef4444;
            font-size: 18px;
            font-weight: 600;
        }}
        
        @media (max-width: 768px) {{
            .error-container {{
                padding: 25px 15px;
            }}
            .error-icon {{
                font-size: 40px;
            }}
            .error-message {{
                font-size: 16px;
            }}
        }}
        
        @media (max-width: 480px) {{
            .error-container {{
                padding: 20px 12px;
                border-radius: 16px;
            }}
            .error-icon {{
                font-size: 36px;
            }}
            .error-message {{
                font-size: 14px;
            }}
        }}
    </style>
    """

# ==========================================
#         FULLY RESPONSIVE GRADIO CSS
# ==========================================
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

* { 
    font-family: 'Inter', sans-serif !important; 
    box-sizing: border-box;
}

body { 
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important; 
    color: #e6edf3 !important; 
}

.gradio-container { 
    max-width: 1800px !important; 
    margin: 0 auto !important;
    padding: 30px !important;
}

/* Header */
.main-header {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
    backdrop-filter: blur(20px);
    border-radius: 20px;
    margin-bottom: 30px;
    border: 2px solid rgba(255,255,255,0.1);
}

.main-header h1 {
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px 0;
}

.main-header p {
    font-size: 18px;
    color: rgba(255,255,255,0.9);
    margin: 0;
}

/* Cards */
.card {
    background: linear-gradient(145deg, rgba(22, 27, 34, 0.8), rgba(28, 33, 40, 0.8)) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px solid rgba(102, 126, 234, 0.2) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
}

.card:hover {
    border-color: rgba(102, 126, 234, 0.4) !important;
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2) !important;
}

.card-title {
    font-size: 18px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #667eea, #a78bfa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 16px !important;
}

/* Button */
button.primary {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
    color: white !important;
    padding: 16px 32px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(34, 197, 94, 0.4) !important;
    cursor: pointer !important;
    width: 100% !important;
    margin: 20px 0 !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #3fb950 0%, #2ea043 100%) !important;
    box-shadow: 0 12px 32px rgba(34, 197, 94, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Accordion */
.accordion {
    background: rgba(22, 27, 34, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px solid rgba(102, 126, 234, 0.2) !important;
    border-radius: 12px !important;
    margin-bottom: 16px !important;
}

/* Image Container */
.image-container {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 2px solid rgba(102, 126, 234, 0.3) !important;
    background: rgba(13, 17, 23, 0.8) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important;
}

.image-container:hover {
    border-color: rgba(102, 126, 234, 0.5) !important;
}

/* Inputs */
input[type="range"] {
    accent-color: #667eea !important;
}

input[type="checkbox"] {
    accent-color: #22c55e !important;
}

label {
    color: rgba(255,255,255,0.9) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 12px; height: 12px; }
::-webkit-scrollbar-track { background: rgba(13, 17, 23, 0.8); border-radius: 8px; }
::-webkit-scrollbar-thumb { 
    background: linear-gradient(180deg, rgba(102, 126, 234, 0.8), rgba(118, 75, 162, 0.8)); 
    border-radius: 8px; 
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #667eea, #764ba2); }

/* Footer */
.footer {
    text-align: center;
    padding: 32px;
    color: rgba(255,255,255,0.6);
    font-size: 14px;
    margin-top: 40px;
    border-top: 2px solid rgba(102, 126, 234, 0.2);
    background: rgba(22, 27, 34, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 16px;
}

/* ============================================ */
/*        RESPONSIVE DESIGN - GRADIO UI         */
/* ============================================ */

/* Tablets (1024px and below) */
@media (max-width: 1024px) {
    .gradio-container { 
        padding: 24px !important; 
        max-width: 100% !important;
    }
    
    .main-header h1 { 
        font-size: 40px !important; 
    }
    
    .main-header p {
        font-size: 16px !important;
    }
    
    .card { 
        padding: 20px !important; 
    }
    
    button.primary {
        padding: 14px 28px !important;
        font-size: 15px !important;
    }
}

/* Tablets Portrait (768px and below) */
@media (max-width: 768px) {
    .gradio-container { 
        padding: 20px !important; 
    }
    
    .main-header {
        padding: 30px 15px !important;
        border-radius: 16px !important;
    }
    
    .main-header h1 { 
        font-size: 32px !important; 
    }
    
    .main-header p {
        font-size: 15px !important;
    }
    
    .card { 
        padding: 16px !important; 
        border-radius: 12px !important;
    }
    
    .card-title {
        font-size: 16px !important;
    }
    
    button.primary {
        padding: 12px 24px !important;
        font-size: 14px !important;
    }
    
    label {
        font-size: 13px !important;
    }
    
    .image-container {
        height: auto !important;
        min-height: 250px !important;
    }
    
    .footer {
        padding: 24px !important;
        font-size: 13px !important;
    }
}

/* Mobile (480px and below) */
@media (max-width: 480px) {
    .gradio-container { 
        padding: 15px !important; 
    }
    
    .main-header {
        padding: 24px 12px !important;
        margin-bottom: 20px !important;
    }
    
    .main-header h1 { 
        font-size: 28px !important; 
    }
    
    .main-header p {
        font-size: 14px !important;
    }
    
    .card { 
        padding: 12px !important; 
        margin-bottom: 15px !important;
    }
    
    .card-title {
        font-size: 15px !important;
        margin-bottom: 12px !important;
    }
    
    button.primary {
        padding: 12px 20px !important;
        font-size: 14px !important;
        margin: 15px 0 !important;
    }
    
    label {
        font-size: 12px !important;
    }
    
    .accordion {
        border-radius: 10px !important;
    }
    
    .image-container {
        border-radius: 10px !important;
        min-height: 200px !important;
    }
    
    .footer {
        padding: 20px !important;
        font-size: 12px !important;
        border-radius: 12px !important;
    }
    
    ::-webkit-scrollbar { 
        width: 8px !important; 
        height: 8px !important; 
    }
}

/* Extra Small Mobile (360px and below) */
@media (max-width: 360px) {
    .gradio-container { 
        padding: 12px !important; 
    }
    
    .main-header h1 { 
        font-size: 24px !important; 
    }
    
    .main-header p {
        font-size: 13px !important;
    }
    
    .card { 
        padding: 10px !important; 
    }
    
    .card-title {
        font-size: 14px !important;
    }
    
    button.primary {
        padding: 10px 16px !important;
        font-size: 13px !important;
    }
    
    label {
        font-size: 11px !important;
    }
}

/* Landscape Mobile */
@media (max-height: 500px) and (orientation: landscape) {
    .main-header {
        padding: 20px 15px !important;
        margin-bottom: 15px !important;
    }
    
    .main-header h1 {
        font-size: 28px !important;
        margin-bottom: 5px !important;
    }
    
    .main-header p {
        font-size: 14px !important;
    }
    
    .image-container {
        min-height: 150px !important;
    }
}
"""

# ==========================================
#     CREATE FULLY RESPONSIVE GRADIO UI
# ==========================================
with gr.Blocks(css=css, title="VisionX Safety - Detection System", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML("""
    <div class='main-header'>
        <h1>🛰️ VisionX Safety</h1>
        <p>AI-Powered Object Detection System</p>
    </div>
    """)
    
    with gr.Row():
        # Left Column
        with gr.Column(scale=1):
            gr.HTML("<div class='card'><div class='card-title'> Upload Image</div></div>")
            input_image = gr.Image(type="pil", label="", elem_classes="image-container", height=350)
            
            with gr.Accordion("🔧 Preprocessing Options", open=False, elem_classes="accordion"):
                preprocessing = gr.Checkbox(label=" Enable Preprocessing", value=False)
                
                brightness_slider = gr.Slider(
                    minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                    label="💡 Brightness"
                )
                
                with gr.Row():
                    contrast_check = gr.Checkbox(label=" Contrast", value=False)
                    denoise_check = gr.Checkbox(label=" Denoise", value=False)
                
                with gr.Row():
                    blur_check = gr.Checkbox(label=" Blur", value=False)
                    sharpen_check = gr.Checkbox(label=" Sharpen", value=False)
                
                edge_check = gr.Checkbox(label=" Edge Detection", value=False)
            
            with gr.Accordion("⚙️ Detection Settings", open=True, elem_classes="accordion"):
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=0.95, value=0.25, step=0.05,
                    label=" Confidence Threshold"
                )
                
                iou_slider = gr.Slider(
                    minimum=0.1, maximum=0.95, value=0.45, step=0.05,
                    label="🔄 IoU Threshold"
                )
        
        # Right Column
        with gr.Column(scale=1):
            gr.HTML("<div class='card'><div class='card-title'> Detection Results</div></div>")
            result_html = gr.HTML()
    
    # Detection Button
    detect_btn = gr.Button(" Run Detection Analysis", variant="primary", elem_classes="primary-button")
    
    # Output Images
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='card'><div class='card-title'> Preprocessed Image</div></div>")
            preprocessed_output = gr.Image(type="pil", label="", height=550, elem_classes="image-container")
        
        with gr.Column(scale=1):
            gr.HTML("<div class='card'><div class='card-title'> Detection Output</div></div>")
            detection_output = gr.Image(type="pil", label="", height=550, elem_classes="image-container")
    
    # Footer
    gr.HTML("""
    <div class='footer'>
        <p style='font-size: 16px; font-weight: 700; margin-bottom: 12px;'>
             VisionX Safety Detection System v2.0
        </p>
        <p>Professional Object Detection System</p>
        <p style='font-size: 12px; margin-top: 16px; opacity: 0.7;'>
            Powered by AI Technology
        </p>
    </div>
    """)
    
    # Events
    detect_btn.click(
        fn=detect_gradio,
        inputs=[
            input_image, preprocessing, contrast_check, blur_check, edge_check, 
            brightness_slider, denoise_check, sharpen_check, conf_slider, iou_slider
        ],
        outputs=[preprocessed_output, detection_output, result_html]
    )
    
    input_image.upload(
        fn=lambda x: create_error_html("✅ Image uploaded! Click 'Run Detection' to analyze.") if x else "",
        inputs=input_image,
        outputs=result_html
    )

# ==========================================
#     START SERVERS
# ==========================================
def start_servers():
    """Start FastAPI and Gradio servers"""
    import threading
    
    print("\n" + "="*70)
    print(" VISIONX SAFETY DETECTION SYSTEM - FULLY RESPONSIVE")
    print("="*70)
    print(f"📁 Model: {'✓ Loaded' if model else '✗ Not Loaded'}")
    if model:
        print(f"🏷️  Classes: {len(model.names)}")
    print("="*70)
    print("🌐 FastAPI:  http://localhost:7000")
    print("🖥️  Gradio:   http://localhost:7860")
    print("📚 API Docs: http://localhost:7000/docs")
    print("="*70)
    print("📱 Responsive: Desktop | Tablet | Mobile")
    print("="*70 + "\n")
    
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=7000, log_level="warning")
    
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    start_servers()