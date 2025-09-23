#!/bin/bash

# Clean startup script for Spyne application
# This script sets all necessary environment variables to suppress warnings

echo "Starting Spyne application with clean environment..."

# Set environment variables to suppress all warnings
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

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "fast_demo.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "python.*server" 2>/dev/null || true

# Wait a moment for processes to die
sleep 1

# Start the backend server
echo "Starting backend server..."
cd "$(dirname "$0")"
python3 -c "import uvicorn; from server import app; uvicorn.run(app, host='0.0.0.0', port=8000)" &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Start the pose detection demo
echo "Starting pose detection (clean mode)..."
python3 fast_demo.py &
DEMO_PID=$!

echo "Application started successfully!"
echo "Backend server PID: $SERVER_PID"
echo "Pose detection PID: $DEMO_PID"
echo ""
echo "To stop the application, run:"
echo "pkill -f 'fast_demo.py' && pkill -f 'uvicorn'"
echo ""
echo "Press Ctrl+C to stop this script (but processes will continue running)"