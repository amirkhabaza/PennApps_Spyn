from __future__ import annotations
import json
import time
import os
from typing import List, Dict, Optional

import requests

# --- Core analyzer comes from your pose_logic.py ---
try:
    from pose_logic import analyze_with_llm as _baseline_analyze_with_llm
except Exception as e:
    _baseline_analyze_with_llm = None
    raise

# -----------------------------
# Gemini (Google Generative AI)
# -----------------------------
try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception:
    genai = None

# Put your key here ONCE so you don't need to input each run.
# (You can also override via env var GEMINI_API_KEY.)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA3xTqUXU7G6FqhmfLgGVWquCgX7lYmpOY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # safe default

if genai and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def gemini_corrections_from_cerebras(
    session_events: List[Dict],
    overall_score: int,
    duration_sec: float,
    good_pct: float,
    max_items: int = 4,
) -> List[str]:
    """
    Feed Cerebras per-call outputs into Gemini to summarize <=4 concrete corrections.
    If Gemini is unavailable, return a deterministic, deduped fallback.
    """
    # Fallback: deterministic top-K unique tips (skip warnings starting with '⚠️')
    def _fallback() -> List[str]:
        seen, tips = set(), []
        for ev in session_events[-80:]:  # last N
            for tip in ev.get("feedback") or []:
                if not isinstance(tip, str) or tip.startswith("⚠️"):
                    continue
                if tip not in seen:
                    seen.add(tip)
                    tips.append(tip)
                if len(tips) >= max_items:
                    break
        return tips or ["Great posture overall."]

    if not (genai and GEMINI_API_KEY):
        return _fallback()

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        compact_events = [
            {
                "t": int(ev.get("t", 0)),
                "score": int(ev.get("score", 0)),
                "status": str(ev.get("status", "")),
                "feedback": [
                    f for f in (ev.get("feedback") or [])
                    if isinstance(f, str) and not f.startswith("⚠️")
                ][:4],
            }
            for ev in session_events[-50:]
        ]

        prompt = (
            "You are a concise posture coach. From the user's session feedback, "
            "output at most 4 concrete corrections, each <= 12 words.\n"
            f"Overall score: {int(overall_score)}\n"
            f"Session duration (sec): {int(duration_sec)}\n"
            f"Good posture time %: {round(good_pct, 1)}\n"
            "Feedback samples (JSON list):\n"
            f"{json.dumps(compact_events, ensure_ascii=False)}\n\n"
            "Return ONLY a JSON list of strings."
        )

        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        data = json.loads(text) if text.startswith("[") else None
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data[:max_items]
    except Exception:
        pass

    return _fallback()


# ----------------------------------------
# Public wrapper to keep fast_demo import
# ----------------------------------------
def analyze_with_llm(category: str, action: str, kps: Dict):
    """
    Thin wrapper so fast_demo imports from api.py remain unchanged.
    Returns: (status, feedback, score, metrics, chosen_action)
    """
    return _baseline_analyze_with_llm(category, action, kps)


# ----------------------------------------
# Real-time session metrics aggregator
# ----------------------------------------
class SessionAggregator:
    """
    Maintains a time-weighted session of scores to compute:
    - overall_score (time-weighted average)
    - session_duration_sec
    - good_posture_pct (time-weighted % with score >= GOOD_CUTOFF)
    - corrections (via Gemini, throttled)
    """

    def __init__(
        self,
        good_cutoff: int = 85,
        metrics_paths: Optional[List[str]] = None,
        gemini_min_interval_sec: float = 10.0,
    ):
        # 🔹 Use absolute path to guarantee metrics.json is always in backend folder
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.good_cutoff = good_cutoff
        self.metrics_paths = metrics_paths or [
            os.path.join(BASE_DIR, "metrics.json"),
            "/tmp/ar_pt_metrics.json",
        ]
        self.gemini_min_interval_sec = gemini_min_interval_sec

        self.reset()

    def reset(self):
        self.session_start: Optional[float] = None
        self.last_event_time: Optional[float] = None
        self.last_score: Optional[int] = None

        self.total_time = 0.0         # seconds (integrated)
        self.score_time_sum = 0.0     # sum(score * dt)
        self.good_time = 0.0          # seconds with score >= good_cutoff

        self.events: List[Dict] = []  # keep recent event summaries
        self._cached_corrections: List[str] = []
        self._last_gemini_ts: float = 0.0

    def update(self, status: str, feedback: List[str], score: int, now: Optional[float] = None) -> Dict:
        now = now or time.time()

        # Start session on first result
        if self.session_start is None:
            self.session_start = now

        # Integrate time using *previous* score over [last_event_time, now]
        if self.last_event_time is not None and self.last_score is not None:
            dt = max(0.0, now - self.last_event_time)
            self.total_time += dt
            self.score_time_sum += self.last_score * dt
            if self.last_score >= self.good_cutoff:
                self.good_time += dt

        # Record current event and set state for next integration step
        evt = {
            "t": now,
            "score": int(score),
            "status": str(status),
            "feedback": list(feedback or []),
        }
        self.events.append(evt)
        if len(self.events) > 240:  # keep ~12 minutes if 3s/call
            self.events = self.events[-240:]

        self.last_event_time = now
        self.last_score = int(score)

        # Compute *display* metrics including the live segment since last event
        return self.current_metrics(now)

    def _live_totals(self, now: Optional[float] = None):
        now = now or time.time()
        if self.session_start is None:
            return 0.0, 0.0, 0.0

        total = self.total_time
        score_sum = self.score_time_sum
        good = self.good_time

        # Add pending time since last event using current score
        if self.last_event_time is not None and self.last_score is not None:
            dt = max(0.0, now - self.last_event_time)
            total += dt
            score_sum += self.last_score * dt
            if self.last_score >= self.good_cutoff:
                good += dt

        return total, score_sum, good

    def _maybe_gemini(self, overall_score: int, duration_sec: float, good_pct: float) -> List[str]:
        # Throttle Gemini calls
        now = time.time()
        if (now - self._last_gemini_ts) < self.gemini_min_interval_sec:
            return self._cached_corrections

        # Need at least two events for a sensible summary
        if len(self.events) < 2:
            return self._cached_corrections

        try:
            tips = gemini_corrections_from_cerebras(self.events, overall_score, duration_sec, good_pct)
            self._cached_corrections = tips[:4]
            self._last_gemini_ts = now
        except Exception:
            # Keep old cache on any error
            pass

        return self._cached_corrections

    def current_metrics(self, now: Optional[float] = None) -> Dict:
        total, score_sum, good = self._live_totals(now)

        duration = float(total)
        if duration > 0:
            overall = int(round(score_sum / duration))
            good_pct = float(min(100.0, max(0.0, (good / duration) * 100.0)))
        else:
            overall = int(self.last_score or 0)
            good_pct = 100.0 if (self.last_score and self.last_score >= self.good_cutoff) else 0.0

        corrections = self._maybe_gemini(overall, duration, good_pct)

        out = {
            "overall_score": overall,
            "session_duration_sec": int(round(duration)),
            "good_posture_pct": round(good_pct, 1),
            "corrections": corrections,
            "last_event": self.events[-1] if self.events else None,
        }

        # Persist to JSON for the website to poll
        self._save_json(out)
        return out

    def _save_json(self, metrics: Dict):
        for path in self.metrics_paths:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
                # Debug: confirm write path
                # print(f"[SessionAggregator] Updated metrics at {path}")
            except Exception as e:
                print(f"[SessionAggregator] Failed to write {path}: {e}")
