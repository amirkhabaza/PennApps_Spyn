#!/bin/bash

# Enhanced Exercise Mode with Selection Requirements
# This script starts the exercise mode with proper selection workflow

echo "🏋️‍♂️ Starting Enhanced Exercise Mode"
echo "===================================="
echo ""
echo "✨ NEW FEATURES:"
echo "  • Exercise selection required before analysis"
echo "  • Real-time form scoring and session time"
echo "  • Voice feedback integration"
echo "  • Professional coaching interface"
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
echo "🧹 Cleaning up..."
pkill -f "fast_demo" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
sleep 1

# Start backend server for voice feedback
echo "🔧 Starting backend server for voice feedback..."
cd "$(dirname "$0")"
python3 -c "import uvicorn; from server import app; uvicorn.run(app, host='0.0.0.0', port=8000)" &
BACKEND_PID=$!

# Wait for backend
sleep 3

echo ""
echo "🎮 EXERCISE MODE CONTROLS:"
echo "=========================="
echo "  2         = Switch to Exercise Mode"
echo "  j/k       = Select Exercise (squat/pushup/bicep_curl)"
echo "  SPACE     = Start/Stop Analysis"
echo "  v         = Voice Feedback"
echo "  r         = Reset Session"
echo "  q         = Quit"
echo ""
echo "📊 DISPLAY FEATURES:"
echo "===================="
echo "  • Real-time form score (0-100%)"
echo "  • Session time tracking (MM:SS)"
echo "  • Exercise-specific feedback"
echo "  • Voice-ready coaching tips"
echo ""
echo "🔊 VOICE FEEDBACK:"
echo "=================="
echo "  • Natural language coaching"
echo "  • ElevenLabs AI voice synthesis"
echo "  • Exercise-specific guidance"
echo "  • Real-time encouragement"
echo ""

echo "🚀 Starting Enhanced Exercise Analysis..."
echo "Focus the control window and follow the prompts!"
echo ""

# Start the enhanced fast_demo
python3 fast_demo.py

# Cleanup
kill $BACKEND_PID 2>/dev/null || true
echo ""
echo "🎉 Enhanced Exercise Session Completed!"
