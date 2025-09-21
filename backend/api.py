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

    Alerts:
      - Keep a *single canonical* `alerts` list (self.current_alerts) that is written
        to metrics.json. Do not duplicate alerts into last_event AND top-level `alerts`.
      - Provide set_no_person() for the "no person detected" case.
      - Deduplicate alerts and apply a short cooldown to avoid spammy repeats.
    """

    def __init__(
        self,
        good_cutoff: int = 85,
        metrics_paths: Optional[List[str]] = None,
        gemini_min_interval_sec: float = 10.0,
    ):
        # Use absolute path to guarantee metrics.json is always in backend folder
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.good_cutoff = good_cutoff
        self.metrics_paths = metrics_paths or [
            os.path.join(BASE_DIR, "metrics.json"),
            "/tmp/ar_pt_metrics.json",
        ]
        self.gemini_min_interval_sec = gemini_min_interval_sec

        self.reset()

    def reset(self):
        """Reset aggregator internal state for a fresh session."""
        self.session_start: Optional[float] = None
        self.last_event_time: Optional[float] = None
        self.last_score: Optional[int] = None

        self.total_time = 0.0         # seconds (integrated)
        self.score_time_sum = 0.0     # sum(score * dt)
        self.good_time = 0.0          # seconds with score >= good_cutoff

        self.events: List[Dict] = []  # keep recent event summaries
        self._cached_corrections: List[str] = []
        self._last_gemini_ts: float = 0.0

        # Correction counter - counts feedback entries only when score < good_cutoff
        self.correction_count: int = 0

        # Alert deduplication + cooldown
        self.current_alerts: List[Dict] = []      # canonical single alert list for UI
        self._recent_alerts: Dict[str, float] = {}  # map alert_key -> timestamp
        self._alert_cooldown_sec: float = 30.0

    # --------------------
    # Alert helpers
    # --------------------
    def _normalize_alerts(self, alerts: Optional[List[Dict]]) -> List[Dict]:
        """
        Deduplicate and normalize alerts list.
        Keying by (type, sorted(parts)). Returns canonical list.
        """
        if not alerts:
            return []
        seen = set()
        out: List[Dict] = []
        for a in alerts:
            if not isinstance(a, dict):
                continue
            t = a.get("type")
            parts = a.get("parts")
            key = (t, tuple(sorted(parts)) if isinstance(parts, list) else None)
            if key in seen:
                continue
            seen.add(key)
            entry = {"type": t}
            if key[1] is not None:
                entry["parts"] = list(key[1])
            out.append(entry)
        return out

    def _should_show_alert(self, alert: Dict, now: float) -> bool:
        """
        Simple cooldown check to avoid repeating identical alerts too frequently.
        Returns True if the alert should be shown now (and records it).
        """
        # Construct stable key
        parts = alert.get("parts") or []
        try:
            key_parts = ",".join(sorted(map(str, parts)))
        except Exception:
            key_parts = str(parts)
        alert_key = f"{alert.get('type', 'unknown')}:{key_parts}"

        # Purge old keys
        cutoff = now - self._alert_cooldown_sec
        self._recent_alerts = {k: v for k, v in self._recent_alerts.items() if v > cutoff}

        if alert_key in self._recent_alerts:
            return False

        # record and allow
        self._recent_alerts[alert_key] = now
        return True

    def set_no_person(self):
        """
        Public helper: set a single 'no_person' alert for the UI when no landmarks are detected.
        Call this from fast_demo/camera loop when you detect no person.
        """
        self.current_alerts = [{"type": "no_person"}]

    # --------------------
    # Main update logic
    # --------------------
    def update(
        self,
        status: str,
        feedback: List[str],
        score: int,
        now: Optional[float] = None,
        metrics: Optional[Dict] = None,
        alerts: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Update aggregator with a new analyzer result.
        - metrics: optional dict returned by analyzer (may contain "missing_parts" and/or "alerts").
        - alerts: explicit alerts list to use (overrides metrics["alerts"] if provided).
        Returns a dict suitable for display and persisted to metrics.json.
        """
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

        # Count corrections only when score < good_cutoff
        if int(score) < self.good_cutoff and feedback:
            self.correction_count += len(feedback)

        # Add missing parts warning to feedback if needed (keeps compatibility with prior console output)
        final_feedback = list(feedback or [])
        if metrics and "missing_parts" in metrics and metrics["missing_parts"]:
            warning_msg = f"⚠️ Missing from camera: {', '.join(metrics['missing_parts'])}"
            # Only add if not already present
            if not any(f.startswith("⚠️") for f in final_feedback):
                final_feedback.insert(0, warning_msg)

        # Build event summary (do NOT duplicate alerts into last_event to avoid frontend double-read)
        evt = {
            "t": now,
            "score": int(score),
            "status": str(status),
            "feedback": final_feedback,
        }

        # Determine alerts to use: explicit alerts param > metrics["alerts"] > derived missing_parts
        alerts_to_use: List[Dict] = []
        if alerts:
            alerts_to_use = list(alerts)
        elif metrics and isinstance(metrics, dict) and metrics.get("alerts"):
            alerts_to_use = list(metrics.get("alerts") or [])
        # if missing_parts present but no explicit alerts, synthesize missing_parts alert
        if metrics and isinstance(metrics, dict) and metrics.get("missing_parts"):
            mp = metrics.get("missing_parts")
            if mp:
                alerts_to_use.append({"type": "missing_parts", "parts": list(mp)})
        
        # Add poor posture alert when score is very low
        if int(score) < 65:  # Poor posture threshold
            alerts_to_use.append({"type": "poor_posture", "score": int(score)})

        # Normalize and apply cooldown dedupe
        normalized = self._normalize_alerts(alerts_to_use)
        filtered = []
        for a in normalized:
            # allow all alerts through normalization; apply cooldown filter to avoid spam
            if self._should_show_alert(a, now):
                filtered.append(a)

        # Update canonical current_alerts (single place for the frontend)
        # If no filtered alerts (e.g., deduped by cooldown), keep previous current_alerts if it is still recent.
        # Here we replace with filtered to reflect current detection; fast_demo can call set_no_person()
        # for "no person" case when there are no landmarks.
        self.current_alerts = filtered

        # Append event summary (we intentionally DO NOT include alerts/missing_parts fields inside evt
        # to avoid duplication in metrics.json; frontend should read `alerts` top-level instead).
        self.events.append(evt)
        if len(self.events) > 240:  # keep ~12 minutes if frequent calls
            self.events = self.events[-240:]

        self.last_event_time = now
        self.last_score = int(score)

        # Compute display metrics including live segment since last event
        return self.current_metrics(now)

    # --------------------
    # Live totals and Gemini summary
    # --------------------
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
        """
        Compute display metrics and persist to JSON. Always include a single canonical 'alerts'
        field (possibly empty list). Do NOT duplicate alerts into last_event and the root.
        """
        total, score_sum, good = self._live_totals(now)

        duration = float(total)
        if duration > 0:
            overall = int(round(score_sum / duration))
            good_pct = float(min(100.0, max(0.0, (good / duration) * 100.0)))
        else:
            overall = int(self.last_score or 0)
            good_pct = 100.0 if (self.last_score and self.last_score >= self.good_cutoff) else 0.0

        corrections = self.correction_count

        out = {
            "overall_score": overall,
            "session_duration_sec": int(round(duration)),
            "good_posture_pct": round(good_pct, 1),
            "corrections": corrections,
            # last_event remains for debugging/console usage but purposely DOES NOT contain 'alerts'
            "last_event": self.events[-1] if self.events else None,
            # canonical alerts list for the frontend to read
            "alerts": list(self.current_alerts or []),
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
