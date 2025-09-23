#!/usr/bin/env python3
"""
Clean wrapper for fast_demo.py that eliminates ALL warnings
This script sets environment variables BEFORE any imports happen
"""

import os
import sys
import warnings
from contextlib import redirect_stderr
import io

# Set ALL environment variables BEFORE any other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_alsologtostderr'] = '0'
os.environ['GLOG_log_dir'] = '/tmp'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION'] = '1'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_V4L2'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '0'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_CAP_AVFOUNDATION'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['URLLIB3_DISABLE_WARNINGS'] = '1'

# Suppress ALL Python warnings
warnings.filterwarnings("ignore")

# Redirect stderr to suppress C++ library warnings
class NullWriter:
    def write(self, txt): pass
    def flush(self): pass

# Temporarily redirect stderr during imports
original_stderr = sys.stderr
sys.stderr = NullWriter()

try:
    # Import everything with suppressed stderr
    import fast_demo
    
    # Restore stderr for application output
    sys.stderr = original_stderr
    
    print("=== AR PT Coach (Clean Mode - No Warnings) ===")
    print("Starting with all warnings suppressed...")
    
    # Run the main function
    fast_demo.main()
    
except KeyboardInterrupt:
    sys.stderr = original_stderr
    print("\nApplication stopped by user.")
except Exception as e:
    sys.stderr = original_stderr
    print(f"Error: {e}")
finally:
    sys.stderr = original_stderr