#  AI Inventory Optimizer

An enterprise-grade intelligent inventory management dashboard that bridges the gap between **Computer Vision (YOLOv8)** and **Neural Time-Series Forecasting (Meta Prophet)**. This system optimizes warehouse stock levels, monitors assets via dual-stream ingestion, and automates supply chain risk mitigation protocols.

---

##  Key Modules & Architecture

* **📷 Module 1: Real-Time Vision Ingestion (YOLOv8)**
  * Supports dual-stream ingestion: **Static Image Manifest Uploads** and **Live Webcam Feed Captures**.
  * Executes real-time tensor inference utilizing cached YOLOv8 weights for optimized memory efficiency.
  * Features a dynamic KPI reporting engine that calculates stock thresholds and triggers automated risk mitigation alerts if reserves fall below baseline safety levels.

* ** Module 2: Predictive Stock Analytics (Meta Prophet)**
  * Processes historical multi-day ingestion pipelines to isolate seasonality trends.
  * Deploys a neural forecasting engine projecting a customizable 5 to 30-day lead-horizon demand vector (including Floor and Ceiling constraints).

---

##  Deep Tech Stack

* **Core Language:** Python
* **Computer Vision Inference:** YOLOv8 (Ultralytics Open-Source Package), OpenCV
* **Predictive Forecasting Engine:** Meta Prophet Model
* **Data Engineering Frameworks:** Pandas, NumPy
* **Enterprise Interface & Caching:** Streamlit Framework (with `@st.cache_resource` optimization)

---

##  Local Installation & Deployment

Follow these structured steps to deploy the application in your local environment:

### 1. Clone the Production Repository
```bash
git clone [https://github.com/pranjalmandhan/AI-Inventory-Optimizer.git](https://github.com/pranjalmandhan/AI-Inventory-Optimizer.git)
cd AI-Inventory-Optimizer
