#!/usr/bin/env python3
"""
GUI Exercise Interface for Spyne Application
Modern button-based interface replacing keyboard controls
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import os
import warnings
import sys
from contextlib import redirect_stderr
import io

# Suppress warnings before imports
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['URLLIB3_DISABLE_WARNINGS'] = '1'

# Import our exercise analysis modules
sys.path.insert(0, os.path.dirname(__file__))
from pose_logic import CATEGORIES, DISPLAY_NAMES, analyze
from api import analyze_with_llm, SessionAggregator
import requests

class ExerciseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spyne Exercise Trainer")
        self.root.geometry("800x600")
        self.root.configure(bg='#2b2b2b')
        
        # Exercise state
        self.current_exercise = None
        self.is_analyzing = False
        self.session_start_time = None
        self.current_form_score = 0
        self.session_aggregator = SessionAggregator(good_cutoff=85)
        self.last_feedback = []
        
        # Camera and pose detection
        self.cap = None
        self.pose = None
        self.analysis_thread = None
        self.running = False
        
        self.setup_gui()
        self.setup_camera()
        
    def setup_gui(self):
        """Create the modern GUI interface"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2b2b2b')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(title_frame, text="🏋️‍♂️ Spyne Exercise Trainer", 
                              font=('Arial', 24, 'bold'), fg='#00ff88', bg='#2b2b2b')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="AI-Powered Form Analysis with Voice Coaching", 
                                 font=('Arial', 12), fg='#888888', bg='#2b2b2b')
        subtitle_label.pack()
        
        # Exercise selection frame
        selection_frame = tk.LabelFrame(self.root, text="Select Exercise", 
                                       font=('Arial', 14, 'bold'), fg='#ffffff', bg='#2b2b2b')
        selection_frame.pack(pady=20, padx=40, fill='x')
        
        # Exercise buttons
        exercise_button_frame = tk.Frame(selection_frame, bg='#2b2b2b')
        exercise_button_frame.pack(pady=15)
        
        self.exercise_buttons = {}
        exercises = CATEGORIES['EXERCISE']
        exercise_info = {
            'squat': {'emoji': '🏋️', 'desc': 'Squat Analysis\nDepth, form, symmetry'},
            'pushup': {'emoji': '💪', 'desc': 'Push-up Analysis\nRange, alignment, balance'},
            'bicep_curl': {'emoji': '💪', 'desc': 'Bicep Curl Analysis\nForm, stability, ROM'}
        }
        
        for i, exercise in enumerate(exercises):
            info = exercise_info.get(exercise, {'emoji': '🏃', 'desc': exercise.title()})
            
            btn_frame = tk.Frame(exercise_button_frame, bg='#2b2b2b')
            btn_frame.grid(row=0, column=i, padx=15)
            
            btn = tk.Button(btn_frame, 
                           text=f"{info['emoji']}\n{exercise.replace('_', ' ').title()}", 
                           font=('Arial', 12, 'bold'),
                           bg='#404040', fg='#ffffff', 
                           activebackground='#00ff88', activeforeground='#000000',
                           width=12, height=3,
                           command=lambda ex=exercise: self.select_exercise(ex))
            btn.pack(pady=5)
            
            desc_label = tk.Label(btn_frame, text=info['desc'], 
                                 font=('Arial', 9), fg='#888888', bg='#2b2b2b')
            desc_label.pack()
            
            self.exercise_buttons[exercise] = btn
        
        # Control buttons frame
        control_frame = tk.LabelFrame(self.root, text="Exercise Controls", 
                                     font=('Arial', 14, 'bold'), fg='#ffffff', bg='#2b2b2b')
        control_frame.pack(pady=20, padx=40, fill='x')
        
        control_button_frame = tk.Frame(control_frame, bg='#2b2b2b')
        control_button_frame.pack(pady=15)
        
        # Start/Stop button
        self.start_stop_btn = tk.Button(control_button_frame, 
                                       text="▶️ Start Analysis", 
                                       font=('Arial', 14, 'bold'),
                                       bg='#00aa00', fg='#ffffff',
                                       activebackground='#00ff00', activeforeground='#000000',
                                       width=15, height=2,
                                       command=self.toggle_analysis,
                                       state='disabled')
        self.start_stop_btn.grid(row=0, column=0, padx=10)
        
        # Voice feedback button
        self.voice_btn = tk.Button(control_button_frame, 
                                  text="🔊 Voice Feedback", 
                                  font=('Arial', 14, 'bold'),
                                  bg='#0066cc', fg='#ffffff',
                                  activebackground='#0088ff', activeforeground='#000000',
                                  width=15, height=2,
                                  command=self.speak_feedback,
                                  state='disabled')
        self.voice_btn.grid(row=0, column=1, padx=10)
        
        # Reset button
        self.reset_btn = tk.Button(control_button_frame, 
                                  text="🔄 Reset Session", 
                                  font=('Arial', 14, 'bold'),
                                  bg='#cc6600', fg='#ffffff',
                                  activebackground='#ff8800', activeforeground='#000000',
                                  width=15, height=2,
                                  command=self.reset_session)
        self.reset_btn.grid(row=0, column=2, padx=10)
        
        # Status and metrics frame
        metrics_frame = tk.LabelFrame(self.root, text="Live Metrics", 
                                     font=('Arial', 14, 'bold'), fg='#ffffff', bg='#2b2b2b')
        metrics_frame.pack(pady=20, padx=40, fill='both', expand=True)
        
        # Metrics display
        metrics_display_frame = tk.Frame(metrics_frame, bg='#2b2b2b')
        metrics_display_frame.pack(fill='both', expand=True, padx=20, pady=15)
        
        # Current status
        self.status_label = tk.Label(metrics_display_frame, 
                                    text="🔴 Select an exercise to begin", 
                                    font=('Arial', 16, 'bold'), 
                                    fg='#ff6666', bg='#2b2b2b')
        self.status_label.pack(pady=10)
        
        # Metrics grid
        metrics_grid = tk.Frame(metrics_display_frame, bg='#2b2b2b')
        metrics_grid.pack(pady=20)
        
        # Form Score
        score_frame = tk.Frame(metrics_grid, bg='#404040', relief='raised', bd=2)
        score_frame.grid(row=0, column=0, padx=15, pady=10, sticky='nsew')
        
        tk.Label(score_frame, text="Form Score", font=('Arial', 12, 'bold'), 
                fg='#ffffff', bg='#404040').pack(pady=5)
        self.score_label = tk.Label(score_frame, text="0%", font=('Arial', 24, 'bold'), 
                                   fg='#00ff88', bg='#404040')
        self.score_label.pack(pady=10)
        
        # Session Time
        time_frame = tk.Frame(metrics_grid, bg='#404040', relief='raised', bd=2)
        time_frame.grid(row=0, column=1, padx=15, pady=10, sticky='nsew')
        
        tk.Label(time_frame, text="Session Time", font=('Arial', 12, 'bold'), 
                fg='#ffffff', bg='#404040').pack(pady=5)
        self.time_label = tk.Label(time_frame, text="00:00", font=('Arial', 24, 'bold'), 
                                  fg='#ffaa00', bg='#404040')
        self.time_label.pack(pady=10)
        
        # Reps Count
        reps_frame = tk.Frame(metrics_grid, bg='#404040', relief='raised', bd=2)
        reps_frame.grid(row=0, column=2, padx=15, pady=10, sticky='nsew')
        
        tk.Label(reps_frame, text="Reps", font=('Arial', 12, 'bold'), 
                fg='#ffffff', bg='#404040').pack(pady=5)
        self.reps_label = tk.Label(reps_frame, text="0", font=('Arial', 24, 'bold'), 
                                  fg='#ff88ff', bg='#404040')
        self.reps_label.pack(pady=10)
        
        # Configure grid weights
        for i in range(3):
            metrics_grid.columnconfigure(i, weight=1)
        
        # Feedback display
        feedback_frame = tk.Frame(metrics_display_frame, bg='#2b2b2b')
        feedback_frame.pack(fill='both', expand=True, pady=20)
        
        tk.Label(feedback_frame, text="💡 Real-time Feedback", 
                font=('Arial', 14, 'bold'), fg='#ffffff', bg='#2b2b2b').pack()
        
        self.feedback_text = tk.Text(feedback_frame, height=6, width=80,
                                    font=('Arial', 11), bg='#1a1a1a', fg='#ffffff',
                                    wrap=tk.WORD, state='disabled')
        self.feedback_text.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Scrollbar for feedback
        scrollbar = tk.Scrollbar(self.feedback_text)
        scrollbar.pack(side='right', fill='y')
        self.feedback_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.feedback_text.yview)
        
        # Initial feedback
        self.update_feedback("Welcome to Spyne Exercise Trainer! Select an exercise above to begin your AI-powered form analysis session.")
        
    def setup_camera(self):
        """Initialize camera and pose detection"""
        try:
            # MediaPipe setup
            mp_pose = mp.solutions.pose
            self.pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                smooth_landmarks=True,
                smooth_segmentation=True,
            )
            
            # Camera setup
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                self.update_feedback("✅ Camera initialized successfully!")
            else:
                self.update_feedback("❌ Failed to initialize camera. Please check your camera connection.")
                
        except Exception as e:
            self.update_feedback(f"❌ Setup error: {str(e)}")
    
    def select_exercise(self, exercise):
        """Handle exercise selection"""
        self.current_exercise = exercise
        
        # Update button states
        for ex, btn in self.exercise_buttons.items():
            if ex == exercise:
                btn.configure(bg='#00ff88', fg='#000000')
            else:
                btn.configure(bg='#404040', fg='#ffffff')
        
        # Enable start button
        self.start_stop_btn.configure(state='normal')
        
        # Update status
        self.status_label.configure(text=f"🟡 {exercise.replace('_', ' ').title()} selected - Ready to start!", 
                                   fg='#ffaa00')
        
        self.update_feedback(f"✅ Selected: {exercise.replace('_', ' ').title()}\n"
                           f"📝 Click 'Start Analysis' to begin real-time form analysis.\n"
                           f"🎯 Position yourself in front of the camera and prepare for the exercise.")
    
    def toggle_analysis(self):
        """Start or stop exercise analysis"""
        if not self.is_analyzing:
            self.start_analysis()
        else:
            self.stop_analysis()
    
    def start_analysis(self):
        """Start exercise analysis"""
        if not self.current_exercise or not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Please select an exercise and ensure camera is working.")
            return
        
        self.is_analyzing = True
        self.session_start_time = time.time()
        self.running = True
        
        # Update UI
        self.start_stop_btn.configure(text="⏸️ Stop Analysis", bg='#cc0000')
        self.voice_btn.configure(state='normal')
        self.status_label.configure(text=f"🟢 Analyzing {self.current_exercise.replace('_', ' ').title()} - Perform exercise now!", 
                                   fg='#00ff88')
        
        # Disable exercise selection during analysis
        for btn in self.exercise_buttons.values():
            btn.configure(state='disabled')
        
        self.update_feedback(f"🏋️‍♂️ ANALYSIS STARTED: {self.current_exercise.replace('_', ' ').title()}\n"
                           f"📊 Real-time form scoring is now active.\n"
                           f"💡 Perform the exercise and watch your form score update live!")
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
        self.analysis_thread.start()
    
    def stop_analysis(self):
        """Stop exercise analysis"""
        self.is_analyzing = False
        self.running = False
        
        # Update UI
        self.start_stop_btn.configure(text="▶️ Start Analysis", bg='#00aa00')
        self.status_label.configure(text=f"🟡 {self.current_exercise.replace('_', ' ').title()} selected - Ready to start!", 
                                   fg='#ffaa00')
        
        # Re-enable exercise selection
        for btn in self.exercise_buttons.values():
            btn.configure(state='normal')
        
        self.update_feedback(f"⏸️ ANALYSIS STOPPED\n"
                           f"📊 Session completed. Click 'Start Analysis' to resume or select a different exercise.")
    
    def analysis_loop(self):
        """Main analysis loop running in separate thread"""
        last_analysis_time = 0
        analysis_cooldown = 0.1  # seconds between analyses
        
        while self.running and self.is_analyzing:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                current_time = time.time()
                
                if results.pose_landmarks and current_time - last_analysis_time >= analysis_cooldown:
                    # Convert landmarks to dict format
                    kps = self.landmarks_to_dict(results.pose_landmarks.landmark)
                    
                    # Perform analysis
                    status, feedback, score, metrics, _ = analyze_with_llm('EXERCISE', self.current_exercise, kps)
                    
                    # Update session aggregator
                    rt = self.session_aggregator.update(status, feedback, score, current_time, metrics)
                    
                    # Update GUI in main thread
                    self.root.after(0, self.update_analysis_results, score, feedback, rt)
                    
                    last_analysis_time = current_time
                
                # Update session time
                if self.session_start_time:
                    session_duration = int(current_time - self.session_start_time)
                    self.root.after(0, self.update_session_time, session_duration)
                
                time.sleep(0.01)  # Limit loop frequency for more responsive updates
                
            except Exception as e:
                print(f"Analysis error: {e}")
                time.sleep(1)
    
    def landmarks_to_dict(self, landmarks):
        """Convert MediaPipe landmarks to dict format"""
        return {
            lm.name: (
                landmarks[lm.value].x,
                landmarks[lm.value].y,
                landmarks[lm.value].z,
                landmarks[lm.value].visibility,
            )
            for lm in mp.solutions.pose.PoseLandmark
        }
    
    def update_analysis_results(self, score, feedback, session_metrics):
        """Update GUI with analysis results"""
        self.current_form_score = score
        self.last_feedback = feedback
        
        # Update score display
        self.score_label.configure(text=f"{score}%")
        
        # Color code the score
        if score >= 85:
            score_color = '#00ff88'  # Green
        elif score >= 70:
            score_color = '#ffaa00'  # Orange
        else:
            score_color = '#ff6666'  # Red
        
        self.score_label.configure(fg=score_color)
        
        # Update reps count
        reps = session_metrics.get('corrections', 0)
        self.reps_label.configure(text=str(reps))
        
        # Update feedback
        feedback_text = f"📊 Form Score: {score}% ({'Excellent' if score >= 85 else 'Good' if score >= 70 else 'Needs Work'})\n\n"
        
        if feedback:
            feedback_text += "💡 Coaching Tips:\n"
            for i, tip in enumerate(feedback[:3], 1):
                feedback_text += f"{i}. {tip}\n"
        else:
            feedback_text += "✅ Great form! Keep it up!"
        
        feedback_text += f"\n📈 Session Stats: Overall {session_metrics.get('overall_score', 0)}% | Reps: {reps}"
        
        self.update_feedback(feedback_text)
    
    def update_session_time(self, duration):
        """Update session time display"""
        minutes, seconds = divmod(duration, 60)
        self.time_label.configure(text=f"{minutes:02d}:{seconds:02d}")
    
    def speak_feedback(self):
        """Send feedback to voice system"""
        if not self.last_feedback:
            messagebox.showinfo("Voice Feedback", "No feedback available yet. Perform exercise to get coaching tips!")
            return
        
        # Format feedback for voice
        main_feedback = self.last_feedback[0] if self.last_feedback else "Good form!"
        voice_text = main_feedback.replace("—", "-").replace("°", " degrees")
        
        if self.current_form_score >= 85:
            voice_message = f"Excellent! {voice_text} Your score is {self.current_form_score} percent."
        elif self.current_form_score >= 70:
            voice_message = f"{voice_text} Current score: {self.current_form_score} percent. Keep improving!"
        else:
            voice_message = f"{voice_text} Score: {self.current_form_score} percent. Focus on form."
        
        # Send to voice API
        try:
            response = requests.post('http://localhost:8000/speak', 
                                   json={'text': voice_message}, 
                                   timeout=5)
            if response.status_code == 200:
                self.update_feedback(f"🔊 Voice: {voice_message}")
            else:
                messagebox.showerror("Voice Error", "Voice system not available. Make sure backend server is running.")
        except Exception as e:
            messagebox.showerror("Voice Error", f"Failed to connect to voice system: {str(e)}")
    
    def reset_session(self):
        """Reset the current session"""
        if messagebox.askyesno("Reset Session", "Are you sure you want to reset the current session? All metrics will be cleared."):
            self.session_aggregator.reset()
            self.session_start_time = time.time() if self.is_analyzing else None
            self.current_form_score = 0
            self.last_feedback = []
            
            # Reset displays
            self.score_label.configure(text="0%", fg='#00ff88')
            self.time_label.configure(text="00:00")
            self.reps_label.configure(text="0")
            
            self.update_feedback("🔄 Session reset! All metrics cleared. Ready for a fresh start.")
    
    def update_feedback(self, text):
        """Update the feedback text display"""
        self.feedback_text.configure(state='normal')
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.insert(tk.END, text)
        self.feedback_text.configure(state='disabled')
        self.feedback_text.see(tk.END)
    
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        self.is_analyzing = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.root.destroy()

def main():
    """Main function to start the GUI application"""
    # Set up clean environment
    os.environ.update({
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'GLOG_minloglevel': '3',
        'GLOG_logtostderr': '0',
        'MEDIAPIPE_DISABLE_GPU': '1',
        'PYTHONWARNINGS': 'ignore',
        'URLLIB3_DISABLE_WARNINGS': '1'
    })
    
    # Create and run the GUI
    root = tk.Tk()
    app = ExerciseGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
