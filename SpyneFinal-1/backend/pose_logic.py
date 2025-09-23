# pose_logic.py
# Unified posture/exercise/sport analysis with clear categories and actions.
# Focus update:
#   - Sitting & Desk posture scoring uses ONLY upper-body features (webcam-friendly)
#   - Robust, weighted, progressive penalties with helpful feedback
#   - Optional LLM (Cerebras) refinement kept but disabled for POSTURE score mixing by default

from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import json
import numpy as np
import mediapipe as mp
import requests
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()
PoseLandmark = mp.solutions.pose.PoseLandmark

# ================================================================
# Cerebras config (use env var in production; avoid hardcoding keys)
# ================================================================
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "") # <-- replace with env var in production
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = "llama3.1-8b"  # small, fast, instruction-tuned model

# ================================================================
# Categories & actions
# ================================================================
CATEGORIES = {
    "POSTURE": ["sitting", "standing", "desk_posture"],   # Everyday posture
    "EXERCISE": ["squat", "pushup", "bicep_curl"]       # Fitness
}

# For UI display
DISPLAY_NAMES = {
    "POSTURE": "POSTURE",
    "EXERCISE": "EXERCISE",
}

# MediaPipe Pose landmark names
LM = {lm.name: lm for lm in PoseLandmark}

# ================================================================
# Score cutoffs
# ================================================================
EXCELLENT_CUTOFF = 95
GOOD_CUTOFF = 85
OK_CUTOFF = 70

# ================================================================
# Geometry helpers
# ================================================================
# --- ADD: direction-invariant line tilt helper (0..90 deg) ---
def _line_tilt_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Direction-invariant tilt of the line segment relative to horizontal.
    Returns an acute angle in [0, 90] degrees, independent of point order.
    0° = perfectly level (good), 90° = vertical line (bad for level checks).
    """
    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
        return np.nan
    dy = abs(p2[1] - p1[1])
    dx = abs(p2[0] - p1[0])
    return float(np.degrees(np.arctan2(dy, dx)))  # 0..90


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Returns the inner angle ABC (in degrees)."""
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _slope_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    """Angle of line p1->p2 vs horizontal, in degrees."""
    dy, dx = p2[1] - p1[1], p2[0] - p1[0]
    return float(np.degrees(np.arctan2(dy, dx)))


def _get_xy(kps: Dict[str, Tuple[float, ...]], name: str) -> np.ndarray:
    """
    Get normalized (x,y) for a named landmark.
    - If landmark is missing or has low visibility (<0.5), return NaNs.
    - Supports both (x,y) and (x,y,z,visibility).
    """
    v = kps.get(name)
    if v is None:
        return np.array([np.nan, np.nan], dtype=float)

    # Handle Mediapipe's (x, y, z, visibility)
    if len(v) >= 4:
        x, y, z, vis = v
        if vis < 0.85 or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):  # Not confidently visible
            return np.array([np.nan, np.nan], dtype=float)
        return np.array([float(x), float(y)], dtype=float)

    # Fallback: just (x, y)
    return np.array([float(v[0]), float(v[1])], dtype=float)



def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance in normalized image space."""
    return float(np.linalg.norm(p1 - p2))


def _angle_to_vertical_deg(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """
    Absolute angle (deg) between vector (p_from -> p_to) and image vertical up.
    0° means perfectly vertical alignment; larger means more deviation.
    """
    v = p_to - p_from
    if np.any(np.isnan(v)):
        return np.nan
    # vertical up vector in image coordinates is (0, -1)
    vert = np.array([0.0, -1.0], dtype=float)
    cosang = np.dot(v, vert) / (np.linalg.norm(v) * np.linalg.norm(vert) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# ================================================================
# Upper-body metrics (webcam-friendly)
# ================================================================
# --- REPLACE THIS FUNCTION ---
def _head_to_shoulder_distance_ratio(kps) -> float:
    """
    Chin-tuck / neck flexion proxy.
    Euclidean distance from NOSE to shoulder-center, normalized by shoulder width.
    Lower = worse (chin closer to shoulders).
    """
    nose = _get_xy(kps, "NOSE")
    lsh, rsh = _get_xy(kps, "LEFT_SHOULDER"), _get_xy(kps, "RIGHT_SHOULDER")
    if np.any(np.isnan(nose)) or np.any(np.isnan(lsh)) or np.any(np.isnan(rsh)):
        return np.nan
    shoulder_width = _dist(lsh, rsh)
    if shoulder_width < 1e-3:
        return np.nan
    shoulder_center = (lsh + rsh) / 2.0
    dist = _dist(nose, shoulder_center)
    return float(dist / (shoulder_width + 1e-8))


def _ear_to_shoulder_distance_ratio(kps) -> float:
    """
    Additional chin-tuck proxy robust to head pitch.
    Average of (LEFT_EAR->LEFT_SHOULDER distance + RIGHT_EAR->RIGHT_SHOULDER distance) / (2 * shoulder width).
    Lower = worse (ears collapsing toward shoulders when head is dropped).
    """
    le, re = _get_xy(kps, "LEFT_EAR"), _get_xy(kps, "RIGHT_EAR")
    lsh, rsh = _get_xy(kps, "LEFT_SHOULDER"), _get_xy(kps, "RIGHT_SHOULDER")
    if np.any(np.isnan(lsh)) or np.any(np.isnan(rsh)):
        return np.nan
    shoulder_width = _dist(lsh, rsh)
    if shoulder_width < 1e-3:
        return np.nan

    dists = []
    if not (np.any(np.isnan(le)) or np.any(np.isnan(lsh))):
        dists.append(_dist(le, lsh))
    if not (np.any(np.isnan(re)) or np.any(np.isnan(rsh))):
        dists.append(_dist(re, rsh))

    if not dists:
        return np.nan
    avg = float(np.mean(dists))
    return float(avg / (shoulder_width + 1e-8))


def _shoulder_level_deg(kps) -> float:
    """Absolute shoulder tilt (0..90 deg) using direction-invariant tilt."""
    l = _get_xy(kps, "LEFT_SHOULDER")
    r = _get_xy(kps, "RIGHT_SHOULDER")
    return _line_tilt_deg(l, r)

# --- REPLACE THIS FUNCTION ---
def _head_roll_deg(kps) -> float:
    """
    Head roll/lateral tilt: tilt between ears (preferred) or eyes (fallback),
    computed as direction-invariant tilt (0..90 deg).
    """
    le, re = _get_xy(kps, "LEFT_EAR"), _get_xy(kps, "RIGHT_EAR")
    tilt = _line_tilt_deg(le, re)
    if not np.isnan(tilt):
        return tilt
    leye, reye = _get_xy(kps, "LEFT_EYE"), _get_xy(kps, "RIGHT_EYE")
    return _line_tilt_deg(leye, reye)



def _neck_vertical_dev_deg(kps) -> float:
    """
    Neck vertical deviation using vector from shoulder_center to nose.
    Only needs head+shoulders (no hips/knees). 0° is ideal (stacked).
    """
    nose = _get_xy(kps, "NOSE")
    lsh, rsh = _get_xy(kps, "LEFT_SHOULDER"), _get_xy(kps, "RIGHT_SHOULDER")
    if np.any(np.isnan(nose)) or np.any(np.isnan(lsh)) or np.any(np.isnan(rsh)):
        return np.nan
    shoulder_center = (lsh + rsh) / 2.0
    return _angle_to_vertical_deg(shoulder_center, nose)


def _forward_head_ratio(kps) -> float:
    """
    Horizontal offset of NOSE from shoulder_center, normalized by shoulder width.
    Smaller = better. Webcam-friendly proxy for forward head posture.
    Returns NaN if shoulders are not reliable.
    """
    nose = _get_xy(kps, "NOSE")
    lsh, rsh = _get_xy(kps, "LEFT_SHOULDER"), _get_xy(kps, "RIGHT_SHOULDER")
    if np.any(np.isnan(nose)) or np.any(np.isnan(lsh)) or np.any(np.isnan(rsh)):
        return np.nan
    shoulder_width = _dist(lsh, rsh)
    if shoulder_width < 1e-3:
        return np.nan
    shoulder_center = (lsh + rsh) / 2.0
    horiz_offset = abs(nose[0] - shoulder_center[0])
    return float(horiz_offset / (shoulder_width + 1e-8))


# ================================================================
# Weighted scoring utilities (progressive penalties)
# ================================================================
def _penalty_low_is_better(value: float, good_max: float, warn_max: float, bad_max: float, weight: float) -> Tuple[float, Optional[str]]:
    """
    For metrics where lower is better (e.g., degrees, ratios).
    - value <= good_max       -> 0 penalty
    - good_max..warn_max      -> linear 0..0.5*weight penalty
    - warn_max..bad_max       -> linear 0.5..1.0*weight penalty
    - value >= bad_max        -> 1.0*weight penalty
    Returns (penalty_points, optional_coaching_tip)
    """
    if np.isnan(value):
        return 0.0, None
    if value <= good_max:
        return 0.0, None
    if value <= warn_max:
        frac = (value - good_max) / max(1e-6, (warn_max - good_max))
        return frac * (0.5 * weight), None
    if value <= bad_max:
        frac = (value - warn_max) / max(1e-6, (bad_max - warn_max))
        return (0.5 + frac * 0.5) * weight, None
    return weight, None


def _collect_feedback(name: str, value: float, thresholds: Tuple[float, float, float], messages: Tuple[str, str, str]) -> Optional[str]:
    """
    Produce short actionable advice for each metric based on value:
    - If value <= good_max: None
    - If value in (good_max..warn_max]: messages[0]
    - If value in (warn_max..bad_max]:  messages[1]
    - If value > bad_max:               messages[2]
    """
    if np.isnan(value):
        return None
    good_max, warn_max, bad_max = thresholds
    if value <= good_max:
        return None
    if value <= warn_max:
        return messages[0]
    if value <= bad_max:
        return messages[1]
    return messages[2]


def _scale_score_from_valid_weights(base_score: float, used_weights: float) -> float:
    """
    If some metrics are NaN and skipped, we rescale so that perfect remains 100.
    We assume total designed weights = 100. If used_weights < 100, scale up.
    """
    used_weights = max(1e-6, min(100.0, used_weights))
    scale = 100.0 / used_weights
    return float(np.clip(base_score * scale, 0.0, 100.0))


# ================================================================
# Sitting / Desk posture (upper-body only)
# ================================================================
def _analyze_posture_upperbody(kps, setting: str):
    """
    Upper-body-only scoring for 'sitting' and 'desk_posture'.
    Uses only webcam-visible landmarks (no hips/knees).
    Metrics:
      - neck_vertical_dev_deg
      - forward_head_ratio
      - shoulder_level_deg
      - head_roll_deg
      - head_to_shoulder_distance_ratio
      - ear_to_shoulder_distance_ratio
    """

    assert setting in ("sitting", "desk_posture")

    # ---------------------------------------------------------
    # Step 1: Check missing landmarks
    # If any required landmark is not detected (NaN),
    # store them in metrics["missing_parts"] for external warning.
    # ---------------------------------------------------------
    required_parts = {
        "LEFT_SHOULDER": "left shoulder",
        "RIGHT_SHOULDER": "right shoulder",
        "NOSE": "head (nose)",
        "LEFT_EAR": "left ear",
        "RIGHT_EAR": "right ear",
    }

    missing_parts = []
    for lm, label in required_parts.items():
        if np.any(np.isnan(_get_xy(kps, lm))):
            missing_parts.append(label)

    # ---------------------------------------------------------
    # Step 2: Compute posture metrics
    # ---------------------------------------------------------
    neck_dev = _neck_vertical_dev_deg(kps)        # degrees
    fwd_head = _forward_head_ratio(kps)           # horizontal offset (nose → shoulder center)
    sh_level = _shoulder_level_deg(kps)           # shoulder tilt in degrees
    head_roll = _head_roll_deg(kps)               # lateral head tilt
    head_to_shoulder = _head_to_shoulder_distance_ratio(kps)   # higher = better
    ear_to_shoulder  = _ear_to_shoulder_distance_ratio(kps)    # higher = better

    # ---------------------------------------------------------
    # Step 3: Weight design (sum = 100)
    # Desk posture: heavier neck/forward penalties
    # Sitting: more balanced
    # ---------------------------------------------------------
    if setting == "desk_posture":
        weights = {
            "neck_dev": 26.0,
            "fwd_head": 26.0,
            "sh_level": 12.0,
            "head_roll": 8.0,
            "hts": 16.0,
            "ets": 12.0,
        }
    else:  # sitting
        weights = {
            "neck_dev": 24.0,
            "fwd_head": 24.0,
            "sh_level": 14.0,
            "head_roll": 8.0,
            "hts": 18.0,
            "ets": 12.0,
        }

    # ---------------------------------------------------------
    # Step 4: Thresholds (good, warning, bad) + coaching messages
    # ---------------------------------------------------------
    neck_thr = (10.0, 18.0, 35.0)
    neck_msgs = (
        "Lift your gaze slightly and lengthen the back of your neck.",
        "Bring head over shoulders; imagine a string pulling the crown upward.",
        "Strong forward head posture—stack head over shoulders."
    )

    fwd_thr = (0.12, 0.25, 0.45)  # more sensitive
    fwd_msgs = (
        "Gently tuck the chin; keep the nose close to shoulder line.",
        "Reduce forward head; bring screen closer to eye level.",
        "Significant forward head—sit tall and pull chin straight back."
    )

    sh_thr = (18.0, 30.0, 45.0)
    sh_msgs = (
        "Keep shoulders level; avoid leaning to one side.",
        "Relax upper traps and lengthen the shorter side.",
        "Pronounced shoulder tilt—reset posture and square your torso."
    )

    roll_thr = (18.0, 30.0, 45.0)
    roll_msgs = (
        "Level your head (avoid mild tilt).",
        "Reduce head tilt; keep ears level.",
        "Strong head tilt—align ears horizontally."
    )

    hts_thr = (0.80, 0.70, 0.55)  # head-to-shoulder, higher = better
    hts_msgs = (
        "Keep space between chin and shoulders.",
        "Lift head slightly; avoid tucking chin too much.",
        "Severe head drop—raise your head and extend the neck."
    )

    ets_thr = (0.70, 0.60, 0.45)  # ear-to-shoulder, higher = better
    ets_msgs = (
        "Maintain gap between ears and shoulders.",
        "Slightly raise head and lengthen neck.",
        "Ears collapsing to shoulders—lift head and extend neck."
    )

    # ---------------------------------------------------------
    # Step 5: Compute penalties and collect feedback
    # ---------------------------------------------------------
    feedback_candidates: List[str] = []
    penalties = []
    used_weight_sum = 0.0

    def apply_penalty(value, thr, w, msgs, label):
        pen, _ = _penalty_low_is_better(value, *thr, weight=w)
        if not np.isnan(value):
            nonlocal used_weight_sum
            used_weight_sum += w
            fb = _collect_feedback(label, value, thr, msgs)
            if fb: 
                feedback_candidates.append(fb)
        return pen if not np.isnan(value) else 0.0

    penalties.append(apply_penalty(neck_dev, neck_thr, weights["neck_dev"], neck_msgs, "neck_dev"))
    penalties.append(apply_penalty(fwd_head, fwd_thr, weights["fwd_head"], fwd_msgs, "fwd_head"))
    penalties.append(apply_penalty(sh_level, sh_thr, weights["sh_level"], sh_msgs, "sh_level"))
    penalties.append(apply_penalty(head_roll, roll_thr, weights["head_roll"], roll_msgs, "head_roll"))

    # Invert ratio metrics (lower ratio = worse)
    penalties.append(apply_penalty(1.0 - head_to_shoulder,
                                   (1.0 - hts_thr[0], 1.0 - hts_thr[1], 1.0 - hts_thr[2]),
                                   weights["hts"], hts_msgs, "head_to_shoulder"))
    penalties.append(apply_penalty(1.0 - ear_to_shoulder,
                                   (1.0 - ets_thr[0], 1.0 - ets_thr[1], 1.0 - ets_thr[2]),
                                   weights["ets"], ets_msgs, "ear_to_shoulder"))

    # ---------------------------------------------------------
    # Step 6: Compute final score
    # ---------------------------------------------------------
    total_penalty = float(sum(penalties))
    score_raw = 100.0 - total_penalty
    
    # Apply scaling first if needed
    if used_weight_sum < 99.0:
        score = _scale_score_from_valid_weights(score_raw, used_weight_sum)
    else:
        score = float(np.clip(score_raw, 0.0, 100.0))
    
    # Apply missing parts penalty AFTER scaling: reduce score based on number of missing parts
    missing_penalty = len(missing_parts) * 15  # 15 points penalty per missing part
    score = max(0, score - missing_penalty)

    # ---------------------------------------------------------
    # Step 7: Build metrics dictionary (add missing_parts separately)
    # ---------------------------------------------------------
    metrics = {
        "neck_vertical_deviation_deg": None if np.isnan(neck_dev) else round(float(neck_dev), 1),
        "forward_head_ratio": None if np.isnan(fwd_head) else round(float(fwd_head), 3),
        "shoulder_level_deg": None if np.isnan(sh_level) else round(float(sh_level), 1),
        "head_roll_deg": None if np.isnan(head_roll) else round(float(head_roll), 1),
        "head_to_shoulder_ratio": None if np.isnan(head_to_shoulder) else round(float(head_to_shoulder), 3),
        "ear_to_shoulder_ratio": None if np.isnan(ear_to_shoulder) else round(float(ear_to_shoulder), 3),
        "used_weight_sum": round(float(used_weight_sum), 1),
        "alerts": []
    }

    if setting in ("sitting", "desk_posture") and missing_parts:
        uniq = sorted(set(missing_parts))
        metrics["alerts"].append({
            "type": "missing_parts",
            "parts": missing_parts
        })

    # ---------------------------------------------------------
    # Step 8: Final feedback and status
    # ---------------------------------------------------------
    feedback = feedback_candidates[:3] if feedback_candidates else ["Upper-body posture looks good."]
    status = "correct" if score >= 85 else ("improvable" if score >= 70 else "poor")

    return status, feedback, int(round(score)), metrics



# ================================================================
# Standing posture (kept, but still upper-body heavy)
# ================================================================
def _analyze_posture_standing(kps):
    """
    We still compute a standing score, with upper-body emphasis (webcam-friendly).
    Lower-body (knees/hips) are deprioritized due to laptop camera limitations.
    """
    # Use the same upper-body metrics as a base
    status, fb, score, metrics = _analyze_posture_upperbody(kps, setting="sitting")  # similar weighting

    # Optionally: tiny bonus if shoulders-to-ears distance symmetrical (stance symmetry proxy)
    lsh, rsh = _get_xy(kps, "LEFT_SHOULDER"), _get_xy(kps, "RIGHT_SHOULDER")
    lear, rear = _get_xy(kps, "LEFT_EAR"), _get_xy(kps, "RIGHT_EAR")
    if not (np.any(np.isnan(lsh)) or np.any(np.isnan(rsh)) or np.any(np.isnan(lear)) or np.any(np.isnan(rear))):
        dL = _dist(lsh, lear)
        dR = _dist(rsh, rear)
        asym = abs(dL - dR)
        # tiny bonus if symmetry is good
        if asym <= 0.02:
            score = min(100, score + 2)
        elif asym >= 0.10:
            score = max(0, score - 2)
    if "alerts" in metrics:
        metrics["alerts"] = [a for a in metrics["alerts"] if a.get("type") != "missing_parts"]
        
    status = "correct" if score >= 85 else ("improvable" if score >= 70 else "poor")
    return status, fb, score, metrics




# ================================================================
# Public entrypoint (baseline heuristic)
# ================================================================
def detect_everyday_action(kps: Dict[str, Tuple[float, float]]) -> str:
    """
    Heuristics for 'sitting', 'standing', 'desk_posture'.
    With only webcam view, we can't rely on knees; use head/shoulder geometry instead:
      - If neck deviation + forward head are high and shoulders level-ish -> likely desk_posture
      - Else if forward head moderate and head/shoulders visible -> sitting
      - Else -> standing (fallback)
    """
    neck_dev = _neck_vertical_dev_deg(kps)
    fwd_head = _forward_head_ratio(kps)
    sh_level = _shoulder_level_deg(kps)

    # If any required metric is NaN, default to 'sitting' (safer for indoor use)
    if np.isnan(neck_dev) or np.isnan(fwd_head) or np.isnan(sh_level):
        return "sitting"

    # Desk-like: pronounced forward head or neck deviation, but shoulders fairly level
    if (fwd_head >= 0.22 or neck_dev >= 18.0) and sh_level <= 12.0:
        return "desk_posture"

    # Otherwise treat as sitting if moderate deviations
    if (0.12 <= fwd_head < 0.30) or (10.0 <= neck_dev < 22.0):
        return "sitting"

    # Fallback
    return "standing"

def _signal_confidence(metrics: Dict[str, float]) -> float:
    """
    Estimate reliability of posture metrics (0..1).
    Higher = more confident the score reflects real posture.
    """
    # Count how many valid metrics we actually have
    valid = 0
    total = 0
    for k, v in metrics.items():
        if k.startswith("used_weight"):  # skip bookkeeping
            continue
        if v is not None:
            total += 1
            # penalize extreme values (likely landmark error)
            if isinstance(v, (int, float)) and not np.isnan(v):
                valid += 1

    if total == 0:
        return 0.0

    frac_valid = valid / total

    # Also scale by used weight fraction if present
    used_w = metrics.get("used_weight_sum", 100.0)
    w_frac = float(used_w) / 100.0

    return max(0.0, min(1.0, 0.5 * frac_valid + 0.5 * w_frac))


def analyze(category: str, action: str, keypoints: Dict[str, Tuple[float, float]]):
    """Main entrypoint. Returns (status, feedback, score, metrics)."""
    cat = (category or "POSTURE").upper()
    if cat not in CATEGORIES:
        cat = "POSTURE"

    if cat == "POSTURE" and (not action or action == "" or action == "auto"):
        action = detect_everyday_action(keypoints)

    acts = CATEGORIES[cat]
    if action not in acts:
        action = acts[0]

    if cat == "POSTURE":
        if action == "sitting":       return _analyze_posture_upperbody(keypoints, "sitting")
        if action == "standing":      return _analyze_posture_standing(keypoints)
        if action == "desk_posture":  return _analyze_posture_upperbody(keypoints, "desk_posture")

    if cat == "EXERCISE":
        return _analyze_exercise(category, action, keypoints)

    if cat == "SPORT":
        return _analyze_sport(category, action, keypoints)

    return "unknown", ["Unsupported action."], 0, {}


# ================================================================
# Exercise analyzers (kept from your logic)
# ================================================================
def _elbow_angle(kps, side: str) -> float:
    shoulder = _get_xy(kps, f"{side}_SHOULDER")
    elbow = _get_xy(kps, f"{side}_ELBOW")
    wrist = _get_xy(kps, f"{side}_WRIST")
    if np.any(np.isnan(shoulder)) or np.any(np.isnan(elbow)) or np.any(np.isnan(wrist)):
        return np.nan
    return _angle_3pts(shoulder, elbow, wrist)


def _angle_hip_line(kps, side: str) -> float:
    """Bodyline proxy for pushup using shoulder-hip-ankle (may be NaN on webcam)."""
    shoulder = _get_xy(kps, f"{side}_SHOULDER")
    hip = _get_xy(kps, f"{side}_HIP")
    ankle = _get_xy(kps, f"{side}_ANKLE")
    if np.any(np.isnan(shoulder)) or np.any(np.isnan(hip)) or np.any(np.isnan(ankle)):
        return np.nan
    return _angle_3pts(shoulder, hip, ankle)

def _elbow_angle(kps, side: str) -> float:
    shoulder = _get_xy(kps, f"{side}_SHOULDER")
    elbow = _get_xy(kps, f"{side}_ELBOW")
    wrist = _get_xy(kps, f"{side}_WRIST")
    if np.any(np.isnan(shoulder)) or np.any(np.isnan(elbow)) or np.any(np.isnan(wrist)):
        return np.nan
    return _angle_3pts(shoulder, elbow, wrist)


def _angle_hip_line(kps, side: str) -> float:
    """Bodyline proxy for pushup using shoulder-hip-ankle (may be NaN on webcam)."""
    shoulder = _get_xy(kps, f"{side}_SHOULDER")
    hip = _get_xy(kps, f"{side}_HIP")
    ankle = _get_xy(kps, f"{side}_ANKLE")
    if np.any(np.isnan(shoulder)) or np.any(np.isnan(hip)) or np.any(np.isnan(ankle)):
        return np.nan
    return _angle_3pts(shoulder, hip, ankle)


def _analyze_exercise_squat(kps):

    knee_L = _angle_3pts(_get_xy(kps, "LEFT_HIP"), _get_xy(kps, "LEFT_KNEE"), _get_xy(kps, "LEFT_ANKLE"))
    knee_R = _angle_3pts(_get_xy(kps, "RIGHT_HIP"), _get_xy(kps, "RIGHT_KNEE"), _get_xy(kps, "RIGHT_ANKLE"))
    back = np.nanmean([
        _angle_3pts(_get_xy(kps, "LEFT_SHOULDER"), _get_xy(kps, "LEFT_HIP"), _get_xy(kps, "LEFT_KNEE")),
        _angle_3pts(_get_xy(kps, "RIGHT_SHOULDER"), _get_xy(kps, "RIGHT_HIP"), _get_xy(kps, "RIGHT_KNEE"))
    ])

    knee = np.nanmean([knee_L, knee_R])
    
    feedback = []
    penalties = []

    # 1. Knee angle scoring (45% weight) - Based on biomechanics research
    if not np.isnan(knee):
        if 85 <= knee <= 95:  # Ideal range (thigh parallel to ground)
            knee_score = 100
        elif 80 <= knee <= 100:  # Acceptable range
            knee_score = 95 - abs(knee - 90) * 0.4
        elif 70 <= knee <= 110:  # Marginal range
            knee_score = 80 - abs(knee - 90) * 0.25
        elif 60 <= knee <= 120:  # Poor range
            knee_score = 65 - abs(knee - 90) * 0.2
        else:  # Unsafe range
            knee_score = max(20, 40 - abs(knee - 90) * 0.15)
        
        if knee_score < 90:
            if knee < 85:
                feedback.append("Deeper squat needed - aim for ~90° at knees (thighs parallel to ground).")
            elif knee > 95:
                feedback.append("Reduce depth slightly - avoid over-squatting below parallel.")
            penalties.append(("knee_angle", 45 * (100 - knee_score) / 100))
    else:
        knee_score = 40  # Unable to detect knee angle
        feedback.append("Unable to detect knee position - ensure full body is visible.")
        penalties.append(("knee_angle", 27))

    # 2. Torso angle scoring (30% weight) - Based on posture stability research
    if not np.isnan(back):
        if back >= 170:  # Ideal upright
            torso_score = 100
        elif back >= 160:  # Acceptable
            torso_score = 95 - (170 - back) * 0.5
        elif back >= 145:  # Marginal
            torso_score = 80 - (160 - back) * 0.4
        elif back >= 130:  # Poor
            torso_score = 65 - (145 - back) * 0.3
        else:  # Excessive forward lean
            torso_score = max(20, 45 - (130 - back) * 0.4)
        
        if torso_score < 90:
            feedback.append("Keep chest up and torso upright - avoid excessive forward lean.")
            penalties.append(("torso_angle", 30 * (100 - torso_score) / 100))
    else:
        torso_score = 50  # Unable to detect torso angle
        penalties.append(("torso_angle", 15))

    # 3. Symmetry scoring (20% weight) - Based on bilateral coordination research
    if not np.isnan(knee_L) and not np.isnan(knee_R):
        balance_diff = abs(knee_L - knee_R)
        if balance_diff <= 3:  # Excellent symmetry
            balance_score = 100
        elif balance_diff <= 6:  # Good symmetry
            balance_score = 95 - balance_diff * 0.5
        elif balance_diff <= 10:  # Acceptable
            balance_score = 85 - balance_diff * 0.4
        elif balance_diff <= 15:  # Marginal
            balance_score = 75 - balance_diff * 0.3
        else:  # Obvious asymmetry
            balance_score = max(30, 65 - balance_diff * 0.25)
        
        if balance_score < 90:
            feedback.append("Balance both legs evenly - ensure symmetrical movement.")
            penalties.append(("balance", 20 * (100 - balance_score) / 100))
    else:
        balance_score = 60  # Unable to detect balance
        penalties.append(("balance", 8))

    # 4. Depth consistency scoring (5% weight) - Based on movement repetition research
    depth_consistency = 90  # Base score
    if not np.isnan(knee) and 85 <= knee <= 95:
        depth_consistency = 100
    elif not np.isnan(knee):
        depth_consistency = max(70, 100 - abs(knee - 90) * 0.3)
    
    if depth_consistency < 95:
        penalties.append(("depth_consistency", 5 * (100 - depth_consistency) / 100))

    # Calculate final score
    total_penalty = sum(penalty[1] for penalty in penalties)
    final_score = max(0, min(100, int(100 - total_penalty)))

    # Status determination - consistent with posture mode
    status = "correct" if final_score >= 85 else ("improvable" if final_score >= 70 else "poor")

    metrics = {
        "knee_angle": None if np.isnan(knee) else round(float(knee), 1),
        "torso_angle": None if np.isnan(back) else round(float(back), 1),
        "balance_diff": None if (np.isnan(knee_L) or np.isnan(knee_R)) else round(float(abs(knee_L - knee_R)), 1),
        "knee_score": round(float(knee_score), 1) if not np.isnan(knee) else None,
        "torso_score": round(float(torso_score), 1) if not np.isnan(back) else None,
        "balance_score": round(float(balance_score), 1) if not (np.isnan(knee_L) or np.isnan(knee_R)) else None,
        "penalties": [{"type": p[0], "amount": round(p[1], 1)} for p in penalties],
        "scientific_basis": {
            "knee_angle_ideal": "90° ± 5° (thighs parallel to ground)",
            "torso_angle_ideal": "≥170° (upright posture)",
            "symmetry_tolerance": "≤3° difference between knees",
            "assessment_method": "Multi-dimensional biomechanical analysis"
        }
    }
    
    if not feedback:
        feedback = ["Excellent squat form with proper biomechanics!"]
    
    return status, feedback, final_score, metrics



def _analyze_exercise_pushup(kps):
    """
    基于科学文献的俯卧撑分析，采用更精确的评分机制
    科学依据：
    - 理想肘关节角度：90° ± 10°（充分下降）
    - 身体直线：肩-髋-踝应保持直线（≥175°）
    - 手臂对称性：左右手臂角度差异应小于8°
    - 稳定性：保持核心稳定，避免身体摇摆
    """
    left_elb = _elbow_angle(kps, "LEFT")
    right_elb = _elbow_angle(kps, "RIGHT")
    hips = np.nanmean([
        _angle_hip_line(kps, "LEFT"),
        _angle_hip_line(kps, "RIGHT")
    ])
    elb = np.nanmean([left_elb, right_elb])

    # Scoring system based on scientific literature
    feedback = []
    penalties = []

    # 1. Elbow angle scoring (40% weight) - Based on biomechanics research
    if not np.isnan(elb):
        if 80 <= elb <= 100:  # Ideal depth (full descent)
            elbow_score = 100
        elif 75 <= elb <= 105:  # Acceptable range
            elbow_score = 95 - abs(elb - 90) * 0.25
        elif 65 <= elb <= 115:  # Marginal range
            elbow_score = 85 - abs(elb - 90) * 0.2
        elif 55 <= elb <= 125:  # Poor range
            elbow_score = 70 - abs(elb - 90) * 0.15
        else:  # Unsafe range
            elbow_score = max(25, 50 - abs(elb - 90) * 0.1)
        
        if elbow_score < 90:
            if elb > 100:
                feedback.append("Lower deeper - aim for ~90° at elbows for full range of motion.")
            elif elb < 80:
                feedback.append("Good depth! Maintain consistent range of motion.")
            penalties.append(("elbow_angle", 40 * (100 - elbow_score) / 100))
    else:
        elbow_score = 35  # Unable to detect elbow angle
        feedback.append("Unable to detect elbow position - ensure arms are visible.")
        penalties.append(("elbow_angle", 26))

    # 2. Body line scoring (35% weight) - Based on posture stability research
    if not np.isnan(hips):
        if hips >= 175:  # Ideal straight line
            bodyline_score = 100
        elif hips >= 170:  # Acceptable
            bodyline_score = 95 - (175 - hips) * 0.4
        elif hips >= 160:  # Marginal
            bodyline_score = 85 - (170 - hips) * 0.3
        elif hips >= 145:  # Poor
            bodyline_score = 70 - (160 - hips) * 0.25
        else:  # Severe bending
            bodyline_score = max(20, 50 - (145 - hips) * 0.3)
        
        if bodyline_score < 90:
            if hips < 170:
                feedback.append("Straighten body line - keep shoulders, hips, and ankles aligned.")
            penalties.append(("bodyline_angle", 35 * (100 - bodyline_score) / 100))
    else:
        bodyline_score = 55  # Unable to detect body line
        penalties.append(("bodyline_angle", 16))

    # 3. Arm symmetry scoring (20% weight) - Based on bilateral coordination research
    if not np.isnan(left_elb) and not np.isnan(right_elb):
        arm_balance_diff = abs(left_elb - right_elb)
        if arm_balance_diff <= 4:  # Excellent symmetry
            arm_balance_score = 100
        elif arm_balance_diff <= 8:  # Good symmetry
            arm_balance_score = 95 - arm_balance_diff * 0.4
        elif arm_balance_diff <= 12:  # Acceptable
            arm_balance_score = 85 - arm_balance_diff * 0.35
        elif arm_balance_diff <= 18:  # Marginal
            arm_balance_score = 75 - arm_balance_diff * 0.3
        else:  # Obvious asymmetry
            arm_balance_score = max(30, 65 - arm_balance_diff * 0.25)
        
        if arm_balance_score < 90:
            feedback.append("Balance both arms equally - ensure symmetrical movement.")
            penalties.append(("arm_balance", 20 * (100 - arm_balance_score) / 100))
    else:
        arm_balance_score = 65  # Unable to detect arm balance
        penalties.append(("arm_balance", 7))

    # 4. Core stability scoring (5% weight) - Based on core muscle activation research
    stability_score = 90  # Base score
    if not np.isnan(elb) and not np.isnan(hips):
        if 80 <= elb <= 100 and hips >= 175:
            stability_score = 100
        elif 75 <= elb <= 105 and hips >= 170:
            stability_score = 95
        elif 70 <= elb <= 110 and hips >= 165:
            stability_score = 85
        else:
            stability_score = 75
    
    if stability_score < 95:
        penalties.append(("stability", 5 * (100 - stability_score) / 100))

    # Calculate final score
    total_penalty = sum(penalty[1] for penalty in penalties)
    final_score = max(0, min(100, int(100 - total_penalty)))

    # Status determination - consistent with posture mode
    status = "correct" if final_score >= 85 else ("improvable" if final_score >= 70 else "poor")

    metrics = {
        "elbow_angle": None if np.isnan(elb) else round(float(elb), 1),
        "bodyline_angle": None if np.isnan(hips) else round(float(hips), 1),
        "arm_balance_diff": None if (np.isnan(left_elb) or np.isnan(right_elb)) else round(float(abs(left_elb - right_elb)), 1),
        "elbow_score": round(float(elbow_score), 1) if not np.isnan(elb) else None,
        "bodyline_score": round(float(bodyline_score), 1) if not np.isnan(hips) else None,
        "arm_balance_score": round(float(arm_balance_score), 1) if not (np.isnan(left_elb) or np.isnan(right_elb)) else None,
        "penalties": [{"type": p[0], "amount": round(p[1], 1)} for p in penalties],
        "scientific_basis": {
            "elbow_angle_ideal": "90° ± 10° (full range of motion)",
            "bodyline_angle_ideal": "≥175° (shoulder-hip-ankle alignment)",
            "symmetry_tolerance": "≤4° difference between arms",
            "assessment_method": "Multi-dimensional biomechanical analysis"
        }
    }
    
    if not feedback:
        feedback = ["Excellent push-up form with proper biomechanics!"]
    
    return status, feedback, final_score, metrics



def _analyze_exercise_bicep(kps):
    """
    Bicep curl analysis (single-frame, webcam-friendly).
    Matches project style: weighted penalties + concise feedback.

    Angle targets (elbow):
      - Bottom (extended): ~150–170°
      - Top (flexed):     ~40–60°
      - Midpoint strength: ~90° (informational; not directly scored here)

    Metrics (weights sum ≈ 100):
      - Elbow deviation to TOP (target ~50°) ..................... 30
      - Elbow deviation to BOTTOM (target ~160°) ................. 20
      - Shoulder sway (upper-arm stability vs vertical) .......... 25
      - ROM proxy (wrist height vs shoulder–hip span; higher=better) 25
    Returns: (status, feedback, score, metrics)
    """

    # ---------------------------------------------------------
    # Step 1: pick visible side (RIGHT preferred)
    # ---------------------------------------------------------
    def _side_ok(side: str) -> bool:
        sh = _get_xy(kps, f"{side}_SHOULDER")
        el = _get_xy(kps, f"{side}_ELBOW")
        wr = _get_xy(kps, f"{side}_WRIST")
        hip = _get_xy(kps, f"{side}_HIP")
        return not (np.any(np.isnan(sh)) or np.any(np.isnan(el)) or
                    np.any(np.isnan(wr)) or np.any(np.isnan(hip)))

    side = "RIGHT" if _side_ok("RIGHT") else ("LEFT" if _side_ok("LEFT") else None)
    if side is None:
        return (
            "improvable",
            ["Show shoulder, elbow, wrist, and hip clearly to analyze curls."],
            70,
            {
                "side_used": None,
                "elbow_angle": None,
                "elbow_dev_top": None,
                "elbow_dev_bottom": None,
                "shoulder_sway_deg": None,
                "rom_norm": None,
                "used_weight_sum": 0.0,
            }
        )

    # Landmarks (chosen side)
    sh = _get_xy(kps, f"{side}_SHOULDER")
    el = _get_xy(kps, f"{side}_ELBOW")
    wr = _get_xy(kps, f"{side}_WRIST")
    hip = _get_xy(kps, f"{side}_HIP")

    # ---------------------------------------------------------
    # Step 2: metrics
    # ---------------------------------------------------------
    elbow = _elbow_angle(kps, side)  # uses your existing helper
    # deviations (lower is better)
    dev_top    = np.nan if np.isnan(elbow) else float(min(abs(elbow - 50.0), 60.0))   # target 40–60°
    dev_bottom = np.nan if np.isnan(elbow) else float(min(abs(elbow - 160.0), 60.0))  # target 150–170°

    # Upper-arm stability: smaller angle = closer to vertical line (pinned)
    shoulder_sway = _angle_to_vertical_deg(sh, el)

    # ROM proxy (higher is better): wrist height vs shoulder–hip span
    if not (np.any(np.isnan(sh)) or np.any(np.isnan(hip)) or np.any(np.isnan(wr))):
        span = abs(sh[1] - hip[1]) + 1e-8
        rom_norm = float((sh[1] - wr[1]) / span)  # higher wrist => larger positive value
    else:
        rom_norm = np.nan

    # ---------------------------------------------------------
    # Step 3: weights, thresholds, coaching
    # ---------------------------------------------------------
    weights = {"dev_top": 30.0, "dev_bottom": 20.0, "sway": 25.0, "rom": 25.0}

    # Progressive penalty thresholds (good, warn, bad)
    top_thr = (12.0, 22.0, 34.0)       # deviation to 50° (lower = better)
    top_msgs = ("Strong top contraction.", "Curl slightly higher for peak squeeze.", "Incomplete top—aim ~50° elbow angle.")

    bot_thr = (10.0, 18.0, 28.0)       # deviation to 160° (lower = better)
    bot_msgs = ("Full extension achieved.", "Extend a touch more (don’t hyperextend).", "Limited bottom—control down to ~160°.")

    sway_thr = (20.0, 30.0, 45.0)      # lower = better
    sway_msgs = ("Upper arm stable.", "Pin elbow closer to your side.", "Too much swing—keep upper arm still.")

    rom_thr = (0.12, 0.07, 0.03)       # higher = better → invert internally
    rom_msgs = ("Great ROM.", "Lift slightly higher at the top.", "Limited ROM—bring wrist higher.")

    # ---------------------------------------------------------
    # Step 4: penalties + feedback
    # ---------------------------------------------------------
    feedback_candidates: List[str] = []
    penalties = []
    used_weight_sum = 0.0

    def add_low(val, thr, w, msgs):
        nonlocal used_weight_sum
        if np.isnan(val):
            return
        used_weight_sum += w
        pen, _ = _penalty_low_is_better(val, *thr, w)
        penalties.append(pen)
        fb = _collect_feedback("", val, thr, msgs)
        if fb:
            feedback_candidates.append(fb)

    # low-is-better metrics
    add_low(dev_top,    top_thr, weights["dev_top"], top_msgs)
    add_low(dev_bottom, bot_thr, weights["dev_bottom"], bot_msgs)
    add_low(shoulder_sway, sway_thr, weights["sway"], sway_msgs)

    # higher-is-better rom_norm → invert
    if not np.isnan(rom_norm):
        inv_val = 1.0 - rom_norm
        inv_thr = (1.0 - rom_thr[0], 1.0 - rom_thr[1], 1.0 - rom_thr[2])
        add_low(inv_val, inv_thr, weights["rom"], rom_msgs)

    # ---------------------------------------------------------
    # Step 5: score, status, metrics
    # ---------------------------------------------------------
    total_penalty = float(sum(penalties))
    score_raw = 100.0 - total_penalty
    score = _scale_score_from_valid_weights(score_raw, used_weight_sum) if used_weight_sum < 99.0 else float(np.clip(score_raw, 0.0, 100.0))
    status = "correct" if score >= 85 else ("improvable" if score >= 70 else "poor")

    metrics = {
        "side_used": side,
        "elbow_angle": None if np.isnan(elbow) else round(float(elbow), 1),
        "elbow_dev_top": None if np.isnan(dev_top) else round(float(dev_top), 1),
        "elbow_dev_bottom": None if np.isnan(dev_bottom) else round(float(dev_bottom), 1),
        "shoulder_sway_deg": None if np.isnan(shoulder_sway) else round(float(shoulder_sway), 1),
        "rom_norm": None if np.isnan(rom_norm) else round(float(rom_norm), 3),
        "used_weight_sum": round(float(used_weight_sum), 1),
    }

    feedback = feedback_candidates[:3] if feedback_candidates else ["Clean curl mechanics."]
    return status, feedback, int(round(score)), metrics

def _analyze_exercise_with_llm(category: str, action: str, kps):
    # First get baseline score
    if action == "squat":
        status, feedback, score, metrics = _analyze_exercise_squat(kps)
    elif action == "pushup":
        status, feedback, score, metrics = _analyze_exercise_pushup(kps)
    elif action == "bicep_curl":
        status, feedback, score, metrics = _analyze_exercise_bicep(kps)
    else:
        return "unknown", ["Unsupported exercise."], 0, {}
    
    # If LLM available, use LLM for scientific optimization
    if CEREBRAS_API_KEY:
        print(f"[DEBUG] Calling LLM for {action}, base_score: {score}")
        try:
            llm_result = llm_refine_exercise_feedback(action, score, metrics, feedback)
            print(f"[DEBUG] LLM result: {llm_result}")
            if llm_result:
                # Dynamic weight: adjust LLM influence based on base score
                if score >= 85:  
                    llm_weight = 0.2
                elif score >= 70:  # Balance weights for medium scores
                    llm_weight = 0.3
                else:  # Increase LLM influence for low scores
                    llm_weight = 0.4
                
                # Merge LLM suggestions with baseline score
                final_score = int((score * (1 - llm_weight) + llm_result["score"] * llm_weight))
                final_score = max(0, min(100, final_score))
                
                # Smart merge feedback: prioritize LLM scientific advice, supplement baseline feedback
                combined_feedback = []
                
                # First add LLM scientific suggestions (max 2)
                for tip in llm_result["advice"][:2]:
                    if tip and tip.strip():
                        combined_feedback.append(tip.strip())
                
                # Then add non-duplicate suggestions from baseline feedback
                for tip in feedback:
                    if tip and tip.strip() and tip.strip() not in combined_feedback and len(combined_feedback) < 4:
                        combined_feedback.append(tip.strip())
                
                # If feedback insufficient, add LLM biomechanical assessment
                if len(combined_feedback) < 2 and llm_result.get("biomechanics_assessment"):
                    combined_feedback.append(f"科学评估: {llm_result['biomechanics_assessment']}")
                
                # Update status - consistent with posture mode
                final_status = "correct" if final_score >= 85 else ("improvable" if final_score >= 70 else "poor")
                
                # Add detailed LLM info to metrics
                metrics["llm_enhanced"] = {
                    "enabled": True,
                    "base_score": score,
                    "llm_score": llm_result["score"],
                    "final_score": final_score,
                    "llm_weight": llm_weight,
                    "confidence": llm_result.get("confidence", 0.75),
                    "biomechanics_assessment": llm_result.get("biomechanics_assessment", ""),
                    "scientific_optimization": True
                }
                
                return final_status, combined_feedback, final_score, metrics
        except Exception as e:
            print(f"LLM exercise enhancement error: {e}")
            # Use baseline score when LLM fails
            pass
    
    # Use baseline score when no LLM or LLM fails
    metrics["llm_enhanced"] = {
        "enabled": False,
        "reason": "LLM unavailable or failed",
        "scientific_optimization": False
    }
    return status, feedback, score, metrics


def llm_refine_exercise_feedback(action: str, base_score: int, metrics: Dict, baseline_feedback: List[str]) -> Optional[Dict]:
    if not CEREBRAS_API_KEY:
        return None

    # Provide scientific basis based on exercise type
    exercise_standards = {
        "squat": {
            "ideal_angles": "Knee: 90°±5° (parallel to ground), Torso: ≥170° (upright)",
            "symmetry": "≤3° difference between knees",
            "key_points": "Depth, torso alignment, knee tracking, symmetry"
        },
        "pushup": {
            "ideal_angles": "Elbow: 90°±10° (full ROM), Bodyline: ≥175° (straight)",
            "symmetry": "≤4° difference between arms", 
            "key_points": "Range of motion, body alignment, arm symmetry, core stability"
        },
        "bicep_curl": {
            "ideal_angles": "Top: ~50°, Bottom: ~160°, Upper arm: stable",
            "symmetry": "≤5° difference between arms",
            "key_points": "Full ROM, elbow stability, controlled tempo"
        }
    }

    standards = exercise_standards.get(action, exercise_standards["squat"])

    system_msg = {
        "role": "system",
        "content": (
            "You are a fitness expert. Always respond with valid JSON only: "
            '{"score": 85, "advice": ["Keep form steady", "Focus on technique"]}. '
            f"Base score: {base_score}. Vary the score significantly (30-100) based on form quality. "
            "Add some randomness to your scoring to reflect real-world variation. "
            "Consider factors like: form consistency, body alignment, movement quality, and effort level."
        )
    }

    # Extract key metrics data
    angle_data = {}
    penalty_data = []
    
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not k.startswith("llm") and not k.startswith("scientific"):
            angle_data[k] = v
        elif k == "penalties" and isinstance(v, list):
            penalty_data = v

    user_payload = {
        "exercise": action,
        "base_score": base_score,
        "angles": angle_data,
        "feedback": baseline_feedback
    }

    payload = {
        "model": CEREBRAS_MODEL,
        "messages": [
            system_msg,
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0.7,
        "max_tokens": 200,
    }
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}

    try:
        print(f"[DEBUG] Sending LLM request for {action}")
        r = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"[DEBUG] LLM raw response: {content}")
        
        # Simple parsing, use base score if JSON parsing fails
        try:
            parsed = json.loads(content)
            score = int(parsed.get("score", base_score))
            advice = parsed.get("advice", baseline_feedback)
            print(f"[DEBUG] Parsed JSON successfully: score={score}")
        except Exception as json_error:
            print(f"[DEBUG] JSON parse failed: {json_error}")
            # When JSON parsing fails, simple analysis based on content
            score = base_score
            advice = baseline_feedback
            
            # If content contains numbers, try to extract as score adjustment
            import re
            numbers = re.findall(r'\d+', content)
            if numbers:
                # Use first number as score adjustment
                score = max(30, min(100, int(numbers[0])))
                print(f"[DEBUG] Extracted score from text: {score}")
        
        # Ensure score is within reasonable range
        score = max(30, min(100, score))
        
        if isinstance(advice, str):
            advice = [advice]

        result = {
            "score": score,
            "advice": advice[:3],
            "assessment": "LLM enhanced analysis"
        }
        print(f"[DEBUG] Final LLM result: {result}")
        return result
    except Exception as e:
        print(f"LLM exercise feedback error: {e}")
        return None


def _analyze_exercise(category: str, action: str, kps):
    """
    主要的exercise分析入口，使用LLM增强版本
    """
    return _analyze_exercise_with_llm(category, action, kps)


# ================================================================
# Sport analyzers (kept from your logic)
# ================================================================
def _analyze_sport_jj(kps):
    """Jumping jack: wrists above head & feet apart."""
    nose_y = _get_xy(kps, "NOSE")[1]
    lw_y, rw_y = _get_xy(kps, "LEFT_WRIST")[1], _get_xy(kps, "RIGHT_WRIST")[1]
    la, ra = _get_xy(kps, "LEFT_ANKLE"), _get_xy(kps, "RIGHT_ANKLE")
    feet_dist = _dist(la, ra)

    score, feedback = 100, []
    if not (lw_y < nose_y and rw_y < nose_y):
        feedback.append("Raise both hands above head.")
        score -= 40
    if feet_dist < 0.15:
        feedback.append("Widen your stance.")
        score -= 20

    return (
        "correct" if score >= 85 else "improvable",
        feedback or ["Good jumping jack position."],
        max(0, min(100, score)),
        {"feet_distance": None if np.isnan(feet_dist) else round(float(feet_dist), 3)}
    )


def _analyze_sport_throw(kps):
    """Throw: check elbow flexion + shoulder level on right arm."""
    elb = _elbow_angle(kps, "RIGHT")
    slope = _shoulder_level_deg(kps)

    score, feedback = 100, []
    if np.isnan(elb) or elb < 70:
        feedback.append("Cock your throwing arm (increase elbow angle).")
        score -= 30
    if np.isnan(slope) or slope > 20:
        feedback.append("Square shoulders toward target.")
        score -= 20

    return (
        "correct" if score >= 85 else "improvable",
        feedback or ["Nice throwing prep."],
        max(0, min(100, score)),
        {
            "elbow_angle": None if np.isnan(elb) else round(float(elb), 1),
            "shoulder_slope": None if np.isnan(slope) else round(float(slope), 1)
        }
    )


def _analyze_sport(category: str, action: str, kps):
    if action == "jumping_jack": return _analyze_sport_jj(kps)
    if action == "throw":        return _analyze_sport_throw(kps)
    return "unknown", ["Unsupported sport."], 0, {}


# ================================================================
# LLM refinement (kept, but no longer flattens POSTURE scores)
# ================================================================
def llm_refine_feedback(category: str,
                        action: str,
                        baseline_score: int,
                        metrics: Dict[str, float],
                        baseline_feedback: List[str],
                        allowed_delta: int = 12) -> Optional[Dict]:
    """
    Query Cerebras to refine score/feedback with controlled influence.
    The model may adjust the score but MUST keep it within ±allowed_delta
    around baseline_score. Also returns optional `confidence` (0..1).
    """
    if not CEREBRAS_API_KEY:
        return None

    system_msg = {
        "role": "system",
        "content": (
            "You are a strict but helpful posture coach. "
            "You must output ONLY valid JSON with keys: "
            "{score:int, advice:list[str], posture:str, confidence:float?}. "
            "Rules:\n"
            f"- Keep score within ±{allowed_delta} of baseline_score.\n"
            "- Score range overall must be 50..100.\n"
            "- Advice must be short, specific, practical (<= 12 words each).\n"
            "- Prefer penalizing forward head, neck flexion, shoulder tilt/roll for desk/sitting.\n"
            "- If metrics indicate clear issues, do NOT inflate score."
        )
    }

    # Include rich context so LLM knows what we already measured
    # inside llm_refine_feedback(...):
    user_payload = {
        "category": category,
        "action": action,
        "baseline_score": baseline_score,
        "metrics": {
            "upper_body_only": True,
            "values": {
            # align names with what you actually put into metrics in _analyze_posture_upperbody
                "neck_vertical_deviation_deg": metrics.get("neck_vertical_deviation_deg"),
                "forward_head_ratio": metrics.get("forward_head_ratio"),
                "shoulder_level_deg": metrics.get("shoulder_level_deg"),
                "head_roll_deg": metrics.get("head_roll_deg"),
            },
            "subscores": metrics.get("subscores"),
            "weights": metrics.get("weights"),
            "alert_level": metrics.get("alert_level"),
        },
        "baseline_feedback": baseline_feedback,
        "allowed_delta": allowed_delta
    }


    payload = {
        "model": CEREBRAS_MODEL,
        "messages": [
            system_msg,
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        "temperature": 0.15,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {CEREBRAS_API_KEY}", "Content-Type": "application/json"}

    try:
        r = requests.post(CEREBRAS_URL, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = json.loads(content)

        # Clamp score inside allowed delta & hard 50..100
        raw = int(parsed.get("score", baseline_score))
        low, high = baseline_score - allowed_delta, baseline_score + allowed_delta
        clamped = max(low, min(high, raw))
        clamped = max(50, min(100, clamped))

        adv = parsed.get("advice") or baseline_feedback
        if isinstance(adv, str):
            adv = [adv]

        out = {
            "score": clamped,
            "advice": adv,
            "posture": parsed.get("posture", action),
            "confidence": float(parsed.get("confidence", 0.6)),  # optional
        }
        return out
    except Exception:
        return None



def analyze_with_llm(category: str, action: str, keypoints: Dict[str, Tuple[float, float]]):
    """
    Baseline heuristic -> (status, feedback, score, metrics)
    + LLM refinement with dynamic weight and dynamic allowed_delta.
    支持POSTURE和EXERCISE模式的LLM增强
    Returns: (final_status, final_feedback, final_score, metrics, chosen_action)
    """
    # For EXERCISE mode, directly use integrated LLM enhanced version
    if category.upper() == "EXERCISE":
        status, feedback, score, metrics = _analyze_exercise_with_llm(category, action, keypoints)
        return status, feedback, score, metrics, action
    
    # For POSTURE mode, use original LLM enhancement logic
    status, feedback, score, metrics = analyze(category, action, keypoints)
    chosen_action = detect_everyday_action(keypoints) if (
        category.upper() == "POSTURE" and (not action or action == "auto")
    ) else action

    # 1) Infer signal confidence from metrics to decide LLM influence
    conf = _signal_confidence(metrics)
    # Allowed score adjustment window (±)
    if conf < 0.50:
        allowed_delta = 20
        llm_weight = 0.55
    elif conf < 0.75:
        allowed_delta = 15
        llm_weight = 0.45
    else:
        allowed_delta = 10
        llm_weight = 0.35

    # 2) Ask LLM with the dynamic window; if it answers, combine
    llm = llm_refine_feedback(category, chosen_action, score, metrics, feedback, allowed_delta=allowed_delta)

    if llm:
        llm_score = int(llm["score"])
        # Final score is weighted fusion; keep inside 50..100
        fused = (1.0 - llm_weight) * score + llm_weight * llm_score
        final_score = int(round(max(50.0, min(100.0, fused))))

        # Merge feedback (keep your alert text at the top if present)
        def _merge_fb(baseline: List[str], llm_adv: List[str], max_n: int = 4) -> List[str]:
            out: List[str] = []
            # Preserve leading alert line(s) if any
            for s in baseline:
                if s.startswith("⚠️"):
                    out.append(s)
            # Add LLM then baseline tips, dedup preserving order
            for s in llm_adv + baseline:
                if s not in out and not s.startswith("⚠️"):
                    out.append(s)
                if len(out) >= max_n:
                    break
            return out or baseline

        final_feedback = _merge_fb(feedback, llm["advice"], max_n=4)

        # Allow LLM to relabel posture
        if llm.get("posture") and category.upper() == "POSTURE":
            chosen_action = llm["posture"]

        # Map to status tiers
        if final_score >= EXCELLENT_CUTOFF:
            final_status = "excellent"
        elif final_score >= GOOD_CUTOFF:
            final_status = "correct"
        elif final_score >= OK_CUTOFF:
            final_status = "improvable"
        else:
            final_status = "poor"

        # Attach LLM diagnostics for UI/telemetry
        metrics["llm"] = {
            "enabled": True,
            "baseline_score": score,
            "llm_score": llm_score,
            "final_score": final_score,
            "llm_weight": llm_weight,
            "allowed_delta": allowed_delta,
            "signal_confidence": round(conf, 3),
            "llm_confidence": float(llm.get("confidence", 0.6)),
        }

        return final_status, final_feedback, final_score, metrics, chosen_action

    # 3) If LLM failed, return baseline
    metrics["llm"] = {
        "enabled": False,
        "baseline_score": score,
        "llm_weight": 0.0,
        "signal_confidence": round(conf, 3),
        "allowed_delta": None,
    }
    return status, feedback, score, metrics, chosen_action
