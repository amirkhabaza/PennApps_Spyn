#!/bin/bash

# Comprehensive stop script for the complete Spyne application

echo "🛑 Stopping Spyne Application"
echo "=============================="

# Stop all related processes
echo "Stopping all processes..."

# Kill Python processes
pkill -f "fast_demo" 2>/dev/null && echo "  ✅ Stopped fast_demo processes"
pkill -f "uvicorn" 2>/dev/null && echo "  ✅ Stopped uvicorn server"
pkill -f "python.*server" 2>/dev/null && echo "  ✅ Stopped Python servers"

# Kill Electron processes
pkill -f "electron" 2>/dev/null && echo "  ✅ Stopped Electron app"

# Wait for processes to die
sleep 2

# Verify processes are stopped
echo ""
echo "Verification:"
if pgrep -f "fast_demo" > /dev/null; then
    echo "  ⚠️  Warning: fast_demo is still running"
else
    echo "  ✅ fast_demo stopped"
fi

if pgrep -f "uvicorn" > /dev/null; then
    echo "  ⚠️  Warning: uvicorn is still running"
else
    echo "  ✅ uvicorn stopped"
fi

if pgrep -f "electron.*fakefrontend" > /dev/null; then
    echo "  ⚠️  Warning: Electron app is still running"
else
    echo "  ✅ Electron app stopped"
fi

# Check if port 8000 is free
if lsof -i :8000 > /dev/null 2>&1; then
    echo "  ⚠️  Warning: Port 8000 is still in use"
else
    echo "  ✅ Port 8000 is free"
fi

echo ""
echo "🎉 All Spyne processes stopped successfully!"
