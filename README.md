# SafeX--yolo
Advanced object detection system using YOLO with interactive UI and preprocessing features.
real-time object detection using YOLO architecture. 
A custom YOLOv8 model that detects industrial safety equipment such as nitrogen tanks, safety kits......
 

**Smart Safety Equipment Detection using YOLOv8 + Gradio**

##  What is this  
SafeX-YOLO is a custom-trained object detection system that identifies industrial safety equipment — like nitrogen tanks, safety kits....

##  Features  
- Detects PPE and industrial safety objects (helmets, tanks, kits, gloves, etc.)  
- Works with images, videos, and webcam feed  
- Real-time inference using YOLOv8  
- Web UI built with HTML, CSS & Gradio  
- Easy deployment via HuggingFace Spaces  

## 🛠 Tech Stack  
- **Python** — core codebase  
- **YOLOv8 (pre-trained / custom-trained model)** — object detection backbone  
- **HTML & CSS** — frontend layout  
- **Gradio** — interactive UI for inference  
- **HuggingFace Spaces** — deployment & hosting  

## 📦 Installation & Running Locally  
```bash
git clone https://github.com/<vaibhav410>/visionx-yolo.git
cd visionx-yolo
pip install -r requirements.txt
python app.py
```  


 ## Live Demo  
Live demo is available here:  
https://huggingface.co/spaces/Suiii0/visionx-yolo  

##  How to Use  
- Upload an image or video OR use webcam feed  
- Click **Detect** — the model will highlight detected safety objects with bounding boxes and labels  

##  Future Work / Improvements  
- Add support for video-stream detection + real-time alerts  
- Expand the safety-objects dataset (e.g. include fire-extinguishers, gas masks)  
- Add logging / report generation for safety compliance  
- Option to export detection results (JSON / CSV)  

##  License  
Specify license here (e.g. MIT, Apache 2.0)  

## 👤 Author  
Vaibhav Kumar — https://github.com/vaibhav410  
