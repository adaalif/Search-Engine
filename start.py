#!/usr/bin/env python3
"""
Simple startup script for Search Engine FastAPI
"""

import subprocess
import sys
import os

def main():
    print("🔍 Search Engine FastAPI")
    print("=" * 30)
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("❌ main.py not found!")
        return
    
    print("🚀 Starting server...")
    print("🌐 Open: http://localhost:8000")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 30)
    
    try:
        # Run the FastAPI server
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
