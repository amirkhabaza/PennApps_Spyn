#!/bin/bash

# GUI Exercise Mode Startup Script
# Modern button-based interface for exercise analysis

echo "🖥️  Starting Spyne GUI Exercise Trainer"
echo "======================================="
echo ""
echo "✨ GUI FEATURES:"
echo "  • Modern button-based interface"
echo "  • Click to select exercises"
echo "  • Visual form scoring display"
echo "  • Real-time session metrics"
echo "  • Voice feedback buttons"
echo "  • Professional dashboard"
echo ""

# Set clean environment variables
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

# Clean up existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "fast_demo" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "exercise_gui" 2>/dev/null || true
sleep 1

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Start backend server for voice feedback
echo "🔧 Starting backend server for voice feedback..."
python3 -c "import uvicorn; from server import app; uvicorn.run(app, host='0.0.0.0', port=8000)" &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

echo ""
echo "🎮 GUI INTERFACE FEATURES:"
echo "=========================="
echo "  🏋️  Exercise Selection Buttons"
echo "     • Squat Analysis"
echo "     • Push-up Analysis" 
echo "     • Bicep Curl Analysis"
echo ""
echo "  🎛️  Control Buttons"
echo "     • Start/Stop Analysis"
echo "     • Voice Feedback"
echo "     • Reset Session"
echo ""
echo "  📊 Live Metrics Dashboard"
echo "     • Real-time Form Score (0-100%)"
echo "     • Session Time (MM:SS)"
echo "     • Rep Counter"
echo "     • Coaching Feedback"
echo ""
echo "  🔊 Voice Integration"
echo "     • ElevenLabs AI Voice"
echo "     • Exercise-specific coaching"
echo "     • Natural language feedback"
echo ""

echo "🚀 Launching GUI Exercise Trainer..."
echo "Click buttons to interact - no keyboard needed!"
echo ""

# Check if tkinter is available
python3 -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: tkinter not available. Installing..."
    # On macOS, tkinter should be included with Python
    echo "Please ensure Python includes tkinter support."
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start the GUI application
python3 exercise_gui.py

# Cleanup when GUI closes
echo ""
echo "🧹 Cleaning up..."
kill $BACKEND_PID 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true

echo "🎉 GUI Exercise Session Completed!"
echo "Thank you for using Spyne Exercise Trainer!"
