# AR PT Coach AI Package 🏋️‍♂️

A posture and exercise feedback system using **MediaPipe Pose** + **Cerebras LLM**.  
It analyzes human pose in real time and provides **scores** and **feedback** for posture, exercises, and simple sports.

---

## ✨ Features
- **Posture mode**: sitting / standing / desk posture  
- **Exercise mode**: squat / push-up / bicep curl  
- **Sport mode**: jumping jack / throw  
- **Scoring system**: heuristic baseline + optional AI refinement (Cerebras)  
- **Manual switching**:  
  - `1` = POSTURE, `2` = EXERCISE, `3` = SPORT  
  - `j` = previous action, `k` = next action  
  - `q` = quit

---

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/<your-username>/AR_PT_Coach_AI_Package.git
cd AR_PT_Coach_AI_Package

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

### 2.Run (console mode, faster, no video output)
python fast_demo.py

AR_PT_Coach_AI_Package/
│── fast_demo.py        # Fast console demo
│── pose_logic.py       # Core scoring + feedback logic
│── api.py              # API integration (if needed)
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
