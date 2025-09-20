# fast_demo.py
# Console + tiny control window. Manual-only action switching.
# Reworked: keep original behavior + real-time session metrics for the website.
# IMPORTANT: Print & append session event ONLY when a NEW LLM call happens.

import cv2
import mediapipe as mp
import time
import numpy as np

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

def draw_control_overlay(category, action, info_lines=None):
    """
    Draw a tiny black window so cv2.waitKey can receive keystrokes.
    Shows current category/action and hotkeys.
    """
    h, w = 170, 680
    img = np.zeros((h, w, 3), dtype=np.uint8)

    def put(y, text):
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    put(24,  f"Mode: {DISPLAY_NAMES.get(category, category)}  |  Action: {action}")
    put(48,  "1=POSTURE   2=EXERCISE")
    put(72,  "j=prev action   k=next action   q=quit")
    put(96,  "r=reset session") 
    if info_lines:
        y = 120
        for line in info_lines[:2]:
            put(y, line)
            y += 22

    cv2.imshow(CONTROL_WIN, img)

# -----------------------------
# Main
# -----------------------------
def main():
    print("=== AR PT Coach (Console Mode, Manual Only) ===")
    print("Focus the small window titled:", CONTROL_WIN)
    print("Keys: 1=POSTURE | 2=EXERCISE | j=prev | k=next | q=quit")

    category = "POSTURE"
    action_idx = 0
    last_query_key = None
    last_llm_time = 0.0
    cached_result = None
    last_no_person_print = 0.0

    # Real-time session aggregator (writes ./metrics.json and /tmp/ar_pt_metrics.json)
    agg = SessionAggregator(good_cutoff=85)

    # MediaPipe Pose (no video rendering)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)  # try to lower capture FPS for CPU relief

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

            # Decide if we should query LLM this iteration
            trigger_now = False
            if res.pose_landmarks:
                kps = landmarks_to_dict(res.pose_landmarks.landmark)

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
                    rt = agg.update(status, feedback, score, now)

                    # 2) concise console print JUST ONCE per LLM call (fixes duplicates)
                    print("==================================================")
                    ts = time.strftime("%H:%M:%S")
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
                # Still keep website metrics ticking (no event)
                agg.current_metrics(now)

            # Show tiny control window so waitKey works
            draw_control_overlay(category, action, info_lines)

            # Key handling (requires the tiny window to be focused)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                category, action_idx = "POSTURE", 0
                print(f"→ Switched to {DISPLAY_NAMES[category]}: {CATEGORIES[category][action_idx]}")
                last_query_key = None
                agg.reset()  # start a fresh session when switching mode (optional)
            elif key == ord('2'):
                category, action_idx = "EXERCISE", 0
                print(f"→ Switched to {DISPLAY_NAMES[category]}: {CATEGORIES[category][action_idx]}")
                last_query_key = None
                agg.reset()
            elif key == ord('j'):
                action_idx = (action_idx - 1) % len(CATEGORIES[category])
                print(f"→ Action: {CATEGORIES[category][action_idx]}")
                last_query_key = None
            elif key == ord('k'):
                action_idx = (action_idx + 1) % len(CATEGORIES[category])
                print(f"→ Action: {CATEGORIES[category][action_idx]}")
                last_query_key = None
            elif key == ord('r'):
                print("→ Reset requested. Confirm? (y/n): ", end="", flush=True)
                confirm = input().strip().lower()

                if confirm == "y":
                    agg.reset()
                    last_query_key = None
                    print("→ Reset confirmed. Pausing for 3 seconds...")
                    time.sleep(3)
                else:
                    print("→ Reset cancelled. Continuing session...")




    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
