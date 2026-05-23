# AI-Inventory-Optimizer

An enterprise-grade intelligent inventory management dashboard that bridges the gap between **Computer Vision** and **Automated Statistical Forecasting** to optimize warehouse stock levels, track assets in real-time, and prevent supply chain stockouts.

---

##  Key Features

* **Real-Time Stock Detection:** Leverages a custom **YOLOv8** pipeline to process warehouse storage images and video feeds, automatically counting inventory units with high precision.
* **Automated Demand Forecasting:** Utilizes Meta's **Prophet Model** to analyze historical stock intake cycles and generate accurate 7-day lead-time demand forecasts.
* **Dynamic Analytics Dashboard:** Built with **Streamlit** to deliver an intuitive UI for logistics tracking, interactive data visualizations, and proactive threshold safety alerts.

---

##  Tech Stack & Architecture

* **Core Language:** Python
* **Computer Vision:** YOLOv8 (Ultralytics), OpenCV
* **Time-Series Forecasting:** Prophet Model
* **Data Engineering & EDA:** Pandas, NumPy
* **Deployment & Interface:** Streamlit Dashboard

---

##  Installation & Local Setup

Follow these simple steps to set up and run the application locally:

### 1. Clone the Repository
```bash
git clone [https://github.com/pranjalmandhan/AI-Inventory-Optimizer.git](https://github.com/pranjalmandhan/AI-Inventory-Optimizer.git)
cd AI-Inventory-Optimizer
