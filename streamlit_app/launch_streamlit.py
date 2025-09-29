#!/usr/bin/env python3
"""
Bond Sector Rotation Strategy - Streamlit Launcher
=================================================

Simple launcher for the Streamlit bond strategy dashboard.
"""

import subprocess
import sys
import importlib.util

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_streamlit():
    """Install Streamlit if not available"""
    try:
        print("Installing Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("Streamlit installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install Streamlit")
        return False

def main():
    """Main launcher function"""
    print("Bond Sector Rotation Strategy - Streamlit Dashboard")
    print("=" * 60)
    
    # Check if streamlit is installed
    if not check_package('streamlit'):
        print("Streamlit not found. Installing...")
        if not install_streamlit():
            print("Please install Streamlit manually: pip install streamlit")
            return
    
    # Check other required packages
    required_packages = ['pandas', 'numpy', 'yfinance', 'plotly']
    missing_packages = [pkg for pkg in required_packages if not check_package(pkg)]
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("All packages installed!")
        except subprocess.CalledProcessError:
            print("Please install packages manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return
    
    print("Launching Streamlit Bond Strategy Dashboard...")
    print("\n" + "=" * 60)
    print("BOND SECTOR ROTATION STRATEGY - STREAMLIT")
    print("=" * 60)
    print("Dashboard will open in your browser automatically")
    print("\nAvailable Bond Strategies:")
    print("   HYG/TLT: High Yield vs Long Treasuries")
    print("   LQD/IEF: Investment Grade vs Intermediate Treasuries")
    print("   JNK/TLT: Junk Bonds vs Long Treasuries")
    print("   AGG/TLT: Aggregate Bonds vs Long Treasuries")
    print("\nFeatures:")
    print("   • Interactive Streamlit interface")
    print("   • Real-time bond strategy analysis")
    print("   • 4-panel technical analysis charts")
    print("   • Performance metrics and current signals")
    print("   • Parameter adjustment controls")
    print("\nPress Ctrl+C in terminal to stop the server")
    print("=" * 60)
    
    # Launch Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "bond_streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n\nStreamlit app stopped by user")
    except Exception as e:
        print(f"Error launching Streamlit: {e}")

if __name__ == "__main__":
    main()