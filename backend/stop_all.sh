#!/bin/bash

# Stop all Spyne application processes
echo "Stopping all Spyne processes..."

# Kill all related processes
pkill -f "fast_demo.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "python.*server" 2>/dev/null || true
pkill -f "electron" 2>/dev/null || true

# Wait for processes to die
sleep 1

# Verify processes are stopped
if pgrep -f "fast_demo.py" > /dev/null; then
    echo "Warning: fast_demo.py is still running"
else
    echo "✓ fast_demo.py stopped"
fi

if pgrep -f "uvicorn" > /dev/null; then
    echo "Warning: uvicorn is still running"
else
    echo "✓ uvicorn stopped"
fi

if pgrep -f "python.*server" > /dev/null; then
    echo "Warning: Python server is still running"
else
    echo "✓ Python server stopped"
fi

echo "All Spyne processes stopped successfully!"

