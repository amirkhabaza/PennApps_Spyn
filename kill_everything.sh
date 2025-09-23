#!/bin/bash

# AGGRESSIVE STOP EVERYTHING SCRIPT
# This script will kill ALL Spyne-related processes using multiple methods

echo "🔥 KILLING ALL SPYNE PROCESSES"
echo "==============================="

# Method 1: Kill by process name patterns
echo "Method 1: Killing by process patterns..."
pkill -f "fast_demo" 2>/dev/null && echo "  ✅ Killed fast_demo processes"
pkill -f "uvicorn" 2>/dev/null && echo "  ✅ Killed uvicorn processes"
pkill -f "python.*server" 2>/dev/null && echo "  ✅ Killed Python server processes"
pkill -f "python.*8000" 2>/dev/null && echo "  ✅ Killed Python port 8000 processes"
pkill -f "electron.*fakefrontend" 2>/dev/null && echo "  ✅ Killed Electron processes"
pkill -f "SpyneFinal" 2>/dev/null && echo "  ✅ Killed SpyneFinal processes"

# Method 2: Kill by port
echo "Method 2: Killing processes using port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "  ✅ Killed processes on port 8000"

# Method 3: Kill by PID (if we can find them)
echo "Method 3: Finding and killing specific PIDs..."
for pid in $(ps aux | grep -E "(fast_demo|uvicorn|python.*server|python.*8000)" | grep -v grep | awk '{print $2}'); do
    kill -9 $pid 2>/dev/null && echo "  ✅ Killed PID $pid"
done

# Method 4: Force kill any remaining Python processes that might be related
echo "Method 4: Force killing suspicious Python processes..."
for pid in $(ps aux | grep python | grep -v grep | awk '{print $2}'); do
    # Check if the process is running our scripts
    if ps -p $pid -o command= | grep -q -E "(fast_demo|server\.py|uvicorn)"; then
        kill -9 $pid 2>/dev/null && echo "  ✅ Force killed Python PID $pid"
    fi
done

# Wait for processes to die
echo "Waiting for processes to die..."
sleep 2

# Verification
echo ""
echo "🔍 VERIFICATION:"
echo "================"

# Check for remaining processes
remaining_processes=$(ps aux | grep -E "(fast_demo|uvicorn|python.*server|python.*8000|electron.*fakefrontend)" | grep -v grep)

if [ -z "$remaining_processes" ]; then
    echo "✅ ALL PROCESSES KILLED SUCCESSFULLY!"
    echo "✅ No Spyne processes are running"
else
    echo "⚠️  WARNING: Some processes may still be running:"
    echo "$remaining_processes"
    echo ""
    echo "🔥 FORCE KILLING REMAINING PROCESSES..."
    echo "$remaining_processes" | awk '{print $2}' | xargs kill -9 2>/dev/null
    sleep 1
fi

# Check port 8000
if lsof -i:8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is still in use"
    lsof -i:8000 | xargs kill -9 2>/dev/null
    echo "✅ Force killed processes on port 8000"
else
    echo "✅ Port 8000 is free"
fi

# Final check
echo ""
echo "📊 FINAL STATUS:"
echo "================"
if ps aux | grep -E "(fast_demo|uvicorn|python.*server|python.*8000|electron.*fakefrontend)" | grep -v grep >/dev/null; then
    echo "❌ SOME PROCESSES STILL RUNNING"
    ps aux | grep -E "(fast_demo|uvicorn|python.*server|python.*8000|electron.*fakefrontend)" | grep -v grep
else
    echo "🎉 ALL SPYNE PROCESSES SUCCESSFULLY KILLED!"
    echo "🎉 System is clean and ready for fresh start"
fi

echo ""
echo "💡 To start fresh, run:"
echo "   ./start_app_clean.sh"
