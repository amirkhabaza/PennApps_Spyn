#!/bin/bash

# Enhanced Exercise Mode Startup Script
# This script starts the application directly in exercise mode with clean environment

echo "🏋️‍♂️ Starting Spyne Exercise Mode (Bug-Free)"
echo "=============================================="

# Set clean environment variables to suppress ALL warnings
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=3
export GLOG_logtostderr=0
export GLOG_alsologtostderr=0
export GLOG_log_dir=/tmp
export MEDIAPIPE_DISABLE_GPU=1
export OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION=1
export OPENCV_VIDEOIO_PRIORITY_MSMF=0
export OPENCV_VIDEOIO_PRIORITY_FFMPEG=0
export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0
export OPENCV_VIDEOIO_PRIORITY_V4L2=0
export OPENCV_VIDEOIO_PRIORITY_DSHOW=0
export OPENCV_VIDEOIO_DEBUG=0
export OPENCV_VIDEOIO_PRIORITY_CAP_AVFOUNDATION=1
export PYTHONWARNINGS=ignore
export URLLIB3_DISABLE_WARNINGS=1

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "fast_demo" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
sleep 1

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting Exercise Mode..."
echo ""
echo "📋 Available Exercises:"
echo "   1. Squat Analysis"
echo "   2. Push-up Analysis"  
echo "   3. Bicep Curl Analysis"
echo ""
echo "🎮 Controls:"
echo "   j/k = Switch between exercises"
echo "   r = Reset session"
echo "   q = Quit"
echo ""
echo "✨ Features:"
echo "   • Real-time pose analysis"
echo "   • Scientific biomechanical scoring"
echo "   • LLM-enhanced feedback"
echo "   • Form correction tips"
echo ""

cd "$SCRIPT_DIR"

# Create a simple exercise starter script
cat > temp_exercise_starter.py << 'EOF'
#!/usr/bin/env python3
import os
import sys

# Set environment variables before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['URLLIB3_DISABLE_WARNINGS'] = '1'

# Import and run fast_demo with exercise mode
sys.path.insert(0, os.path.dirname(__file__))

# Import fast_demo module
import fast_demo

# Modify the main function to start in exercise mode
def exercise_main():
    print("=== AR PT Coach (Exercise Mode - Clean) ===")
    print("Starting directly in EXERCISE mode...")
    
    # Call the original main but start in exercise mode
    original_main = fast_demo.main
    
    # Monkey patch to start in exercise mode
    def patched_main():
        # Set up the original function
        import cv2
        import mediapipe as mp
        import time
        import numpy as np
        from fast_demo import landmarks_to_dict, suppress_stderr, suppress_all_warnings, draw_control_overlay
        from pose_logic import CATEGORIES, DISPLAY_NAMES
        from api import analyze_with_llm, SessionAggregator
        
        # Suppress warnings
        stderr_file = suppress_all_warnings()
        
        print("Focus the small window titled: AR PT Coach — Controls (focus here)")
        print("Keys: j=prev exercise | k=next exercise | q=quit | r=reset")
        
        category = "EXERCISE"  # Start in exercise mode
        action_idx = 0
        last_query_key = None
        last_llm_time = 0.0
        cached_result = None
        last_no_person_print = 0.0
        
        # Session aggregator
        agg = SessionAggregator(good_cutoff=85)
        
        # MediaPipe setup
        mp_pose = mp.solutions.pose
        with suppress_stderr():
            pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                smooth_landmarks=True,
                smooth_segmentation=True,
            )
        
        # Camera setup
        cap = None
        with suppress_stderr():
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    print("Using AVFoundation backend")
                else:
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        print("Using default backend")
            except:
                cap = cv2.VideoCapture(0)
        
        if not cap or not cap.isOpened():
            print("Cannot open camera.")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        cv2.namedWindow("AR PT Coach — Controls (focus here)")
        
        print(f"→ Starting in {DISPLAY_NAMES[category]}: {CATEGORIES[category][action_idx]}")
        
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                
                actions = CATEGORIES[category]
                action = actions[action_idx % len(actions)]
                
                info_lines = []
                key_now = (category, action)
                now = time.time()
                
                trigger_now = False
                if res.pose_landmarks:
                    kps = landmarks_to_dict(res.pose_landmarks.landmark)
                    
                    if key_now != last_query_key:
                        trigger_now = True
                    elif now - last_llm_time >= 3:  # 3 second cooldown
                        trigger_now = True
                    
                    if trigger_now:
                        status, feedback, score, metrics, _ignored = analyze_with_llm(category, action, kps)
                        cached_result = (status, feedback or [], int(score), metrics or {})
                        last_llm_time = now
                        last_query_key = key_now
                        
                        rt = agg.update(status, feedback, score, now, metrics)
                        
                        print("=" * 60)
                        ts = time.strftime("%H:%M:%S")
                        print(f"[{ts}] Exercise: {action.upper()}")
                        print(f"Score: {score} ({status})")
                        
                        for tip in feedback:
                            if not tip.startswith("⚠️"):
                                print(f" • {tip}")
                        
                        mm = rt.get("session_duration_sec", 0) // 60
                        ss = rt.get("session_duration_sec", 0) % 60
                        print(f"→ Session: {mm:02d}:{ss:02d} | Overall: {rt.get('overall_score', 0)}% | Reps: {rt.get('corrections', 0)}")
                        
                        info_lines.append(f"Score: {score}% | Session: {rt.get('overall_score', 0)}%")
                    else:
                        rt = agg.current_metrics(now)
                        info_lines.append(f"Score: {cached_result[2] if cached_result else 0}% | Session: {rt.get('overall_score', 0)}%")
                else:
                    if now - last_no_person_print > 1.0:
                        print("=" * 60)
                        print("No person detected - position yourself in front of camera")
                        last_no_person_print = now
                    info_lines.append("No person detected")
                    agg.set_no_person()
                    rt = agg.current_metrics(now)
                
                draw_control_overlay(category, action, info_lines)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('j'):
                    action_idx = (action_idx - 1) % len(CATEGORIES[category])
                    print(f"→ Exercise: {CATEGORIES[category][action_idx].upper()}")
                    last_query_key = None
                elif key == ord('k'):
                    action_idx = (action_idx + 1) % len(CATEGORIES[category])
                    print(f"→ Exercise: {CATEGORIES[category][action_idx].upper()}")
                    last_query_key = None
                elif key == ord('r'):
                    print("→ Reset session? (y/n): ", end="", flush=True)
                    confirm = input().strip().lower()
                    if confirm == "y":
                        agg.reset()
                        last_query_key = None
                        print("→ Session reset! Starting fresh...")
                        time.sleep(2)
                    else:
                        print("→ Continuing current session...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if 'stderr_file' in locals():
                try:
                    sys.stderr.close()
                    sys.stderr = sys.__stderr__
                    os.unlink(stderr_file)
                except:
                    pass
    
    patched_main()

if __name__ == "__main__":
    exercise_main()
EOF

echo "Starting exercise mode..."
python3 temp_exercise_starter.py

# Clean up
rm -f temp_exercise_starter.py

echo ""
echo "🎉 Exercise session completed!"
