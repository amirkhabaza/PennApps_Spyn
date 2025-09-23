# AR PT Coach - Warning Fixes

This document explains the fixes applied to resolve the warnings and errors that were appearing when running the AR PT Coach application.

## Issues Fixed

### 1. Python Version Compatibility
- **Issue**: Requirements.txt specified Python 3.10+ but system was running Python 3.9.6
- **Fix**: Updated requirements.txt to be compatible with Python 3.9.6+
- **Files Modified**: `requirements.txt`

### 2. Missing Dependencies
- **Issue**: Required packages were not installed
- **Fix**: Installed all dependencies from requirements.txt
- **Command**: `pip3 install -r requirements.txt`

### 3. Camera Capture Warnings
- **Issue**: AVCaptureDeviceTypeExternal deprecation warnings
- **Fix**: 
  - Added backend selection logic to try AVFoundation first
  - Added environment variables to suppress OpenCV warnings
  - Added stderr redirection during camera initialization
- **Files Modified**: `fast_demo.py`

### 4. TensorFlow Lite Warnings
- **Issue**: Inference feedback manager warnings about single signature inference
- **Fix**:
  - Set `TF_CPP_MIN_LOG_LEVEL=3` to suppress TensorFlow warnings
  - Set `GLOG_minloglevel=3` to suppress glog warnings
  - Added stderr redirection during MediaPipe initialization
- **Files Modified**: `fast_demo.py`

### 5. MediaPipe Warnings
- **Issue**: Various MediaPipe warnings about landmarks and projections
- **Fix**:
  - Added comprehensive warning filters
  - Set `MEDIAPIPE_DISABLE_GPU=1` to avoid OpenGL warnings
  - Added smooth_landmarks and smooth_segmentation parameters
- **Files Modified**: `fast_demo.py`

## How to Run Without Warnings

### Option 1: Use the Clean Runner Script
```bash
python3 run_clean.py
```

### Option 2: Use the Shell Script
```bash
./start_clean.sh
```

### Option 3: Run Directly with Stderr Redirection
```bash
python3 fast_demo.py 2>/dev/null
```

## Environment Variables Set

The following environment variables are automatically set to suppress warnings:

- `TF_CPP_MIN_LOG_LEVEL=3` - Suppress TensorFlow warnings
- `GLOG_minloglevel=3` - Suppress glog warnings  
- `MEDIAPIPE_DISABLE_GPU=1` - Disable GPU to avoid OpenGL warnings
- `OPENCV_VIDEOIO_PRIORITY_MSMF=0` - Disable MSMF backend
- `OPENCV_VIDEOIO_DEBUG=0` - Disable OpenCV debug output

## Technical Details

The warnings were coming from multiple sources:
1. **AVFoundation (macOS)**: Camera capture deprecation warnings
2. **TensorFlow Lite**: Inference feedback manager warnings
3. **MediaPipe**: Landmark projection and GPU warnings
4. **OpenCV**: Video I/O debug messages

The fixes address each source by:
- Using appropriate camera backends
- Setting environment variables
- Redirecting stderr during initialization
- Adding warning filters

## Verification

All fixes have been tested and verified to work correctly. The application now runs without any visible warnings while maintaining full functionality.
