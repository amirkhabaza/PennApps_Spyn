# fast_demo.py
# Console + tiny control window. Manual-only action switching.
# Reworked: keep original behavior + real-time session metrics for the website.
# IMPORTANT: Print & append session event ONLY when a NEW LLM call happens.

import cv2
import mediapipe as mp
import time
import numpy as np
import os
import warnings
import sys
from contextlib import redirect_stderr
import io

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
warnings.filterwarnings("ignore", category=FutureWarning, module="mediapipe")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
warnings.filterwarnings("ignore", message=".*inference_feedback_manager.*")
warnings.filterwarnings("ignore", message=".*NORM_RECT without IMAGE_DIMENSIONS.*")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings
os.environ['GLOG_minloglevel'] = '3'  # Suppress glog warnings
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Disable GPU to avoid OpenGL warnings
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # Disable MSMF backend
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'  # Disable OpenCV debug output
os.environ['OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION'] = '1'  # Use AVFoundation on macOS
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '0'  # Disable FFmpeg
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'  # Disable GStreamer
os.environ['OPENCV_VIDEOIO_PRIORITY_V4L2'] = '0'  # Disable V4L2
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '0'  # Disable DirectShow
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # Disable MSMF
os.environ['OPENCV_VIDEOIO_PRIORITY_CAP_AVFOUNDATION'] = '1'  # Prefer AVFoundation
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Disable GPU completely
os.environ['GLOG_logtostderr'] = '0'  # Don't log to stderr
os.environ['GLOG_log_dir'] = '/tmp'  # Log to temp directory
os.environ['GLOG_alsologtostderr'] = '0'  # Don't also log to stderr
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress all Python warnings
os.environ['URLLIB3_DISABLE_WARNINGS'] = '1'  # Disable urllib3 warnings

from pose_logic import CATEGORIES, DISPLAY_NAMES  # manual categories/actions
from api import analyze_with_llm, SessionAggregator  # <-- aggregator for website metrics

# -----------------------------
# Settings
# -----------------------------
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
LLM_COOLDOWN = 0.1  # seconds between Cerebras calls for same (category, action)

CONTROL_WIN = "AR PT Coach — Controls (focus here)"

# -----------------------------
# Helpers
# -----------------------------
def landmarks_to_dict(landmarks):
    """
    Convert MediaPipe landmarks to {NAME: (x, y, z, visibility)} dict.
    This allows downstream code to check visibility and drop NaNs when part is off-camera.
    """
    return {
        lm.name: (
            landmarks[lm.value].x,
            landmarks[lm.value].y,
            landmarks[lm.value].z,
            landmarks[lm.value].visibility,
        )
        for lm in mp.solutions.pose.PoseLandmark
    }

def suppress_stderr():
    """Context manager to suppress stderr output."""
    return redirect_stderr(io.StringIO())

def suppress_all_warnings():
    """Suppress all possible warnings and logs."""
    # Redirect stderr to devnull
    import subprocess
    import tempfile
    
    # Create a temporary file for stderr
    stderr_file = tempfile.NamedTemporaryFile(delete=False)
    stderr_file.close()
    
    # Redirect stderr to the temporary file
    sys.stderr = open(stderr_file.name, 'w')
    
    return stderr_file.name

def send_voice_feedback(feedback_text):
    """Send feedback to voice system via API call."""
    try:
        import requests
        response = requests.post('http://localhost:8000/speak', 
                               json={'text': feedback_text}, 
                               timeout=5)
        if response.status_code == 200:
            print(f"🔊 Voice feedback sent: {feedback_text}")
        else:
            print(f"Voice feedback failed: {response.status_code}")
    except Exception as e:
        print(f"Voice feedback error: {e}")

def format_voice_feedback(feedback_list, score, exercise_name):
    """Format feedback for voice output - make it natural and concise."""
    if not feedback_list:
        return f"Good {exercise_name} form! Score: {score} percent."
    
    # Take the most important feedback and make it voice-friendly
    main_feedback = feedback_list[0] if feedback_list else ""
    
    # Clean up feedback for voice
    voice_text = main_feedback.replace("—", "-").replace("°", " degrees")
    
    # Add score context
    if score >= 85:
        return f"Excellent! {voice_text} Your score is {score} percent."
    elif score >= 70:
        return f"{voice_text} Current score: {score} percent. Keep improving!"
    else:
        return f"{voice_text} Score: {score} percent. Focus on form."

def draw_control_overlay(category, action, info_lines=None, exercise_selected=False, session_time=0, form_score=None):
    """
    Draw a tiny black window so cv2.waitKey can receive keystrokes.
    Shows current category/action and hotkeys.
    Enhanced for exercise mode with form score and session time.
    """
    h, w = 200, 750
    img = np.zeros((h, w, 3), dtype=np.uint8)

    def put(y, text, color=(0, 255, 0)):
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    put(24,  f"Mode: {DISPLAY_NAMES.get(category, category)}  |  Action: {action}")
    
    if category == "EXERCISE":
        put(48,  "1=POSTURE   2=EXERCISE   SPACE=Start Analysis   v=Voice Feedback")
        put(72,  "j=prev exercise   k=next exercise   q=quit   r=reset")
        
        # Exercise status
        if exercise_selected and form_score is not None:
            mm, ss = divmod(session_time, 60)
            put(96,  f"ANALYZING: Form Score: {form_score}% | Session: {mm:02d}:{ss:02d}", (0, 255, 255))
        elif exercise_selected:
            put(96,  "READY - Press SPACE to start analysis", (255, 255, 0))
        else:
            put(96,  "SELECT EXERCISE - Press j/k to choose, then SPACE to start", (255, 100, 100))
    else:
        put(48,  "1=POSTURE   2=EXERCISE")
        put(72,  "j=prev action   k=next action   q=quit")
        put(96,  "r=reset session")
    
    if info_lines:
        y = 120
        for line in info_lines[:3]:
            put(y, line)
            y += 20

    cv2.imshow(CONTROL_WIN, img)

# -----------------------------
# Main
# -----------------------------
def main():
    # Suppress all warnings and stderr output
    stderr_file = suppress_all_warnings()
    
    # Check command line arguments for mode
    import sys
    start_mode = "POSTURE"  # default
    if len(sys.argv) > 1:
        if sys.argv[1] == "exercise":
            start_mode = "EXERCISE"
        elif sys.argv[1] == "posture":
            start_mode = "POSTURE"
    
    print("=== AR PT Coach (Console Mode, Manual Only) ===")
    print("Focus the small window titled:", CONTROL_WIN)
    print("Keys: 1=POSTURE | 2=EXERCISE | j=prev | k=next | q=quit")
    print(f"Starting in {start_mode} mode")

    category = start_mode
    action_idx = 0
    last_query_key = None
    last_llm_time = 0.0
    cached_result = None
    last_no_person_print = 0.0
    
    # Exercise mode variables
    exercise_selected = False
    exercise_analyzing = False
    session_start_time = time.time()
    current_form_score = None
    last_voice_feedback = ""
    voice_enabled = False
    
    # If starting in exercise mode, automatically select first exercise
    if category == "EXERCISE":
        exercise_selected = True
        print(f"→ Exercise mode: {CATEGORIES[category][action_idx]} selected")
        print("→ Press SPACE to start analysis")

    # Real-time session aggregator (writes ./metrics.json and /tmp/ar_pt_metrics.json)
    agg = SessionAggregator(good_cutoff=85)

    # MediaPipe Pose (no video rendering) - suppress warnings during initialization
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

    # Try different camera backends to avoid AVCaptureDeviceTypeExternal warnings
    cap = None
    with suppress_stderr():
        # Try AVFoundation first (best for macOS)
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                print("Using AVFoundation backend")
            else:
                cap.release()
                cap = None
        except:
            cap = None
            
        # Fallback to default if AVFoundation fails
        if cap is None:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    print("Using default backend")
            except:
                cap = None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)  # try to lower capture FPS for CPU relief
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce buffer size to minimize latency

    if not cap.isOpened():
        print("Cannot open camera.")
        return

    cv2.namedWindow(CONTROL_WIN)

    print(f"→ Start in {DISPLAY_NAMES[category]}: {CATEGORIES[category][action_idx]}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            # Run pose in RGB (not displayed)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # Determine current (manual) action
            actions = CATEGORIES[category]
            action = actions[action_idx % len(actions)]

            info_lines = []
            key_now = (category, action)
            now = time.time()

            # Exercise mode logic - only analyze if exercise is selected and analyzing
            trigger_now = False
            if res.pose_landmarks:
                kps = landmarks_to_dict(res.pose_landmarks.landmark)

                # For exercise mode, only analyze if exercise is selected and analyzing
                if category == "EXERCISE":
                    if exercise_selected and exercise_analyzing:
                        if key_now != last_query_key:
                            trigger_now = True  # immediate call on mode/action change
                        elif now - last_llm_time >= LLM_COOLDOWN:
                            trigger_now = True
                    else:
                        # Not analyzing, just show selection status
                        pass
                else:
                    # Posture mode - normal behavior
                    if key_now != last_query_key:
                        trigger_now = True  # immediate call on mode/action change
                    elif now - last_llm_time >= LLM_COOLDOWN:
                        trigger_now = True

                if trigger_now:
                    # === NEW RESULT ARRIVED: call Cerebras, update session ONCE, print ONCE ===
                    status, feedback, score, metrics, _ignored = analyze_with_llm(category, action, kps)
                    cached_result = (status, feedback or [], int(score), metrics or {})
                    last_llm_time = now
                    last_query_key = key_now

                    # 1) aggregator.append ONE event + compute display metrics
                    rt = agg.update(status, feedback, score, now, metrics)

                    # 2) Enhanced console print for exercise mode
                    print("=" * 60)
                    ts = time.strftime("%H:%M:%S")
                    
                    if category == "EXERCISE":
                        current_form_score = score
                        session_duration = int(now - session_start_time)
                        mm, ss = divmod(session_duration, 60)
                        
                        print(f"[{ts}] EXERCISE: {action.upper()}")
                        print(f"Form Score: {score}% ({status.upper()})")
                        print(f"Session Time: {mm:02d}:{ss:02d}")
                        
                        # Exercise-specific feedback
                        for tip in feedback:
                            if not tip.startswith("⚠️"):
                                print(f" 💡 {tip}")
                        
                        # Store voice-ready feedback
                        last_voice_feedback = format_voice_feedback(feedback, score, action)
                        
                        print(f"→ Session Stats: Overall {rt.get('overall_score', 0)}% | "
                              f"Reps: {rt.get('corrections', 0)} | "
                              f"Duration: {mm:02d}:{ss:02d}")
                        print(f"🔊 Voice ready: Press 'v' for audio feedback")
                        
                        # Overlay for exercise mode
                        info_lines.append(f"Form: {score}% | Session: {rt.get('overall_score', 0)}%")
                        info_lines.append(f"Time: {mm:02d}:{ss:02d} | Reps: {rt.get('corrections', 0)}")
                        
                    else:
                        # Original posture mode display
                        print(f"[{ts}] Mode={DISPLAY_NAMES[category]} / Action={action}")
                        print(f"Score: {score} ({status})")

                        # Special warning for missing body parts (if provided by pose_logic)
                        if metrics and "missing_parts" in metrics and metrics["missing_parts"]:
                            print(f"⚠️ Missing from camera: {', '.join(metrics['missing_parts'])}")

                        for tip in feedback:
                            if not tip.startswith("⚠️"):
                                print(f" - {tip}")

                        mm = rt.get("session_duration_sec", 0) // 60
                        ss = rt.get("session_duration_sec", 0) % 60
                        print(f"→ Overall Score: {rt.get('overall_score', 0)}% | "
                              f"Duration: {mm:02d}:{ss:02d} | "
                              f"Good Posture: {rt.get('good_posture_pct', 0.0)}% | "
                              f"Corrections: {rt.get('corrections', 0)}")

                        # Overlay summary
                        info_lines.append(f"Overall {rt.get('overall_score', 0)}% | "
                                          f"Good {rt.get('good_posture_pct', 0.0)}%")
                else:
                    # === NO NEW RESULT: do NOT append event or print ===
                    # Only refresh metrics.json with live totals for the website
                    rt = agg.current_metrics(now)
                    info_lines.append(f"Overall {rt.get('overall_score', 0)}% | "
                                      f"Good {rt.get('good_posture_pct', 0.0)}%")

            else:
                # Throttle "no person" prints
                if now - last_no_person_print > 1.0:
                    print("==================================================")
                    print("No person detected.")
                    last_no_person_print = now
                info_lines.append("No person detected.")

                # ✅ Let aggregator handle the no_person alert
                agg.set_no_person()
                rt = agg.current_metrics(now)

            # Show enhanced control window with exercise status
            session_duration = int(now - session_start_time) if category == "EXERCISE" else 0
            draw_control_overlay(category, action, info_lines, 
                               exercise_selected, session_duration, current_form_score)

            # Enhanced key handling for exercise mode
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                category, action_idx = "POSTURE", 0
                exercise_selected = False
                exercise_analyzing = False
                print(f"→ Switched to {DISPLAY_NAMES[category]}: {CATEGORIES[category][action_idx]}")
                last_query_key = None
                agg.reset()
            elif key == ord('2'):
                category, action_idx = "EXERCISE", 0
                exercise_selected = False  # Reset selection when entering exercise mode
                exercise_analyzing = False
                session_start_time = time.time()  # Reset session timer
                print(f"→ Switched to {DISPLAY_NAMES[category]}: {CATEGORIES[category][action_idx]}")
                print("→ Select an exercise with j/k keys, then press SPACE to start analysis")
                last_query_key = None
                agg.reset()
            elif key == ord('j'):
                action_idx = (action_idx - 1) % len(CATEGORIES[category])
                current_exercise = CATEGORIES[category][action_idx]
                
                if category == "EXERCISE":
                    exercise_selected = True
                    exercise_analyzing = False  # Stop analyzing when switching exercises
                    print(f"→ Exercise Selected: {current_exercise.upper()}")
                    print("→ Press SPACE to start analysis")
                else:
                    print(f"→ Action: {current_exercise}")
                last_query_key = None
            elif key == ord('k'):
                action_idx = (action_idx + 1) % len(CATEGORIES[category])
                current_exercise = CATEGORIES[category][action_idx]
                
                if category == "EXERCISE":
                    exercise_selected = True
                    exercise_analyzing = False  # Stop analyzing when switching exercises
                    print(f"→ Exercise Selected: {current_exercise.upper()}")
                    print("→ Press SPACE to start analysis")
                else:
                    print(f"→ Action: {current_exercise}")
                last_query_key = None
            elif key == ord(' '):  # Space bar to start/stop exercise analysis
                if category == "EXERCISE" and exercise_selected:
                    exercise_analyzing = not exercise_analyzing
                    if exercise_analyzing:
                        session_start_time = time.time()  # Reset timer when starting
                        print(f"🏋️‍♂️ STARTING ANALYSIS: {CATEGORIES[category][action_idx].upper()}")
                        print("→ Perform the exercise - real-time feedback will appear")
                        print("→ Press SPACE again to stop, 'v' for voice feedback")
                    else:
                        print(f"⏸️  STOPPED ANALYSIS: {CATEGORIES[category][action_idx].upper()}")
                        print("→ Press SPACE to resume analysis")
                elif category == "EXERCISE":
                    print("⚠️  Select an exercise first using j/k keys")
            elif key == ord('v'):  # Voice feedback
                if category == "EXERCISE" and last_voice_feedback and voice_enabled:
                    print(f"🔊 Speaking: {last_voice_feedback}")
                    send_voice_feedback(last_voice_feedback)
                elif category == "EXERCISE" and last_voice_feedback:
                    voice_enabled = True
                    print(f"🔊 Voice enabled! Speaking: {last_voice_feedback}")
                    send_voice_feedback(last_voice_feedback)
                elif category == "EXERCISE":
                    print("⚠️  No feedback available yet - perform exercise to get feedback")
                else:
                    print("⚠️  Voice feedback only available in exercise mode")
            elif key == ord('r'):
                print("→ Reset requested. Confirm? (y/n): ", end="", flush=True)
                confirm = input().strip().lower()

                if confirm == "y":
                    agg.reset()
                    last_query_key = None
                    if category == "EXERCISE":
                        session_start_time = time.time()
                        exercise_analyzing = False
                        print("→ Exercise session reset! Select exercise and press SPACE to start.")
                    else:
                        print("→ Reset confirmed. Pausing for 3 seconds...")
                        time.sleep(3)
                else:
                    print("→ Reset cancelled. Continuing session...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Clean up stderr redirection
        if 'stderr_file' in locals():
            try:
                sys.stderr.close()
                sys.stderr = sys.__stderr__  # Restore original stderr
                os.unlink(stderr_file)  # Delete temporary file
            except:
                pass


if __name__ == "__main__":
    main()
