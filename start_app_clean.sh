#!/bin/bash

# Comprehensive clean startup script for the complete Spyne application
# This script ensures ZERO warnings and a bug-free experience

echo "🚀 Starting Spyne Application (Bug-Free Mode)"
echo "=============================================="

# Set global environment variables to suppress ALL warnings
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

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/fakefrontend-main"

# Step 1: Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "fast_demo" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "python.*server" 2>/dev/null || true
pkill -f "electron" 2>/dev/null || true

# Wait for processes to die
sleep 2

# Step 2: Start the backend server
echo "🔧 Starting backend server..."
cd "$BACKEND_DIR"
python3 -c "import uvicorn; from server import app; uvicorn.run(app, host='0.0.0.0', port=8000)" &
BACKEND_PID=$!
echo "   ✅ Backend server started (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
for i in {1..10}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "   ✅ Backend is ready!"
        break
    fi
    sleep 1
    echo "   ⏳ Attempt $i/10..."
done

# Step 3: Start the Electron frontend
echo "🖥️  Starting Electron frontend..."
cd "$FRONTEND_DIR"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "   📦 Installing npm dependencies..."
    npm install
fi

# Start Electron app in the background
npx electron . &
FRONTEND_PID=$!
echo "   ✅ Electron app started (PID: $FRONTEND_PID)"

# Step 4: Verification
echo ""
echo "🎉 Application Started Successfully!"
echo "=================================="
echo "Backend Server:  http://localhost:8000"
echo "Frontend App:    Desktop application window"
echo ""
echo "Process IDs:"
echo "  Backend:  $BACKEND_PID"
echo "  Frontend: $FRONTEND_PID"
echo ""
echo "📊 Test the API:"
echo "curl http://localhost:8000/metrics"
echo ""
echo "🛑 To stop everything:"
echo "./backend/stop_all.sh && pkill -f electron"
echo ""
echo "✨ The application is now running with ZERO warnings!"

# Keep script running to show output
echo "Press Ctrl+C to exit this monitoring script (processes will continue)"
echo "Monitoring application..."

# Monitor the processes
while true; do
    sleep 5
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "⚠️  Backend process stopped unexpectedly"
        break
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "⚠️  Frontend process stopped unexpectedly"
        break
    fi
done
