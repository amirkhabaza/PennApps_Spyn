#!/usr/bin/env python3
"""
Wrapper script to start fast_demo in exercise mode automatically.
This script starts fast_demo.py and automatically switches to exercise mode.
"""

import subprocess
import time
import sys
import os

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fast_demo_path = os.path.join(script_dir, 'fast_demo.py')
    
    print("Starting fast_demo in exercise mode...")
    
    # Start fast_demo.py
    process = subprocess.Popen([sys.executable, fast_demo_path], 
                             cwd=script_dir,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    
    try:
        # Wait a moment for the process to start
        time.sleep(2)
        
        # Send '2' to switch to exercise mode
        print("Switching to exercise mode...")
        process.stdin.write('2\n')
        process.stdin.flush()
        
        # Let the process run
        print("Fast demo is now running in exercise mode.")
        print("The control window should show 'EXERCISE' mode.")
        print("Press Ctrl+C to stop.")
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping fast_demo...")
        process.terminate()
        process.wait()
        print("Fast demo stopped.")
    except Exception as e:
        print(f"Error: {e}")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()

