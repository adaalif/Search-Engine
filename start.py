#!/usr/bin/env python3
"""
Startup script for Cluster-then-Search Engine FastAPI
"""

import subprocess
import sys
import os

def main():
    print("🔍 Cluster-then-Search Information Retrieval Engine")
    print("=" * 50)
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("❌ main.py not found!")
        return
    
    print("🚀 Starting server...")
    print("🌐 Open: http://localhost:8000")
    print("📊 Features:")
    print("   ✅ Typo correction")
    print("   ✅ Cluster filtering")
    print("   ✅ Multi-field search (title + abstract)")
    print("   ✅ RRF ranking")
    print("   ✅ Dataset selection (CACM, CISI, Inspec)")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Run the FastAPI server
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()