#!/bin/bash
# Quick stop command - one liner
pkill -9 -f "fast_demo" && pkill -9 -f "uvicorn" && pkill -9 -f "python.*server" && pkill -9 -f "electron.*fakefrontend" && lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "✅ All processes killed" || echo "✅ No processes to kill"
