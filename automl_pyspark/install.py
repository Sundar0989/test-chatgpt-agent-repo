#!/usr/bin/env python3
"""
AutoML PySpark Pipeline Installation Script

This script helps you install the right dependencies based on your environment
and use case. It provides different installation options for different scenarios.
"""

import subprocess
import sys
import os
from typing import List

def run_command(command: str) -> bool:
    """Run a shell command and return success status."""
    try:
        print(f"🔄 Running: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running: {command}")
        print(f"   Error output: {e.stderr}")
        return False

def install_packages(packages: List[str], description: str = "") -> bool:
    """Install a list of packages."""
    if description:
        print(f"\n📦 Installing {description}...")
    
    for package in packages:
        if not run_command(f"pip install {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
            return False
    return True

def check_java():
    """Check if Java is installed (required for PySpark)."""
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Java is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️ Java not found. PySpark requires Java 8 or 11.")
    print("   Please install Java: https://adoptopenjdk.net/")
    return False

def install_development():
    """Install minimal dependencies for development."""
    packages = [
        "pyspark>=3.3.0",
        "pandas>=1.3.0", 
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0.0",
        "joblib>=1.1.0",
        "matplotlib>=3.5.0"
    ]
    return install_packages(packages, "development dependencies")

def install_staging():
    """Install balanced dependencies for staging."""
    # First install core
    if not install_development():
        return False
    
    # Then add staging-specific packages
    packages = [
        "optuna>=3.0.0",
        "xgboost>=1.6.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
        "xlsxwriter>=3.0.0"
    ]
    return install_packages(packages, "staging dependencies")

def install_production():
    """Install all dependencies for production."""
    # First install staging
    if not install_staging():
        return False
    
    # Then add production-specific packages
    packages = [
        # "synapseml>=0.11.0",  # Commented out as it can be problematic
    ]
    
    print("\n🚀 Production installation complete!")
    print("📝 Optional packages you may want to install manually:")
    print("   • SynapseML (LightGBM): pip install synapseml>=0.11.0")
    print("   • Streamlit UI: pip install streamlit>=1.20.0")
    print("   • Cloud integrations: pip install google-cloud-bigquery boto3")
    
    return True

def install_full():
    """Install everything including optional packages."""
    if not install_production():
        return False
    
    packages = [
        "streamlit>=1.20.0",
        "jupyter>=1.0.0",
        "plotly>=5.0.0"
    ]
    return install_packages(packages, "full feature set")

def install_dev_tools():
    """Install development tools."""
    packages = [
        "pytest>=7.0.0",
        "black>=22.0.0", 
        "flake8>=5.0.0",
        "mypy>=0.991",
        "isort>=5.10.0"
    ]
    return install_packages(packages, "development tools")

def main():
    """Main installation interface."""
    print("🚀 AutoML PySpark Pipeline Installation")
    print("=" * 50)
    
    # Check prerequisites
    if not check_java():
        choice = input("\nContinue anyway? (y/N): ").lower()
        if choice != 'y':
            sys.exit(1)
    
    print("\nSelect installation type:")
    print("1. 🔧 Development (minimal, fast setup)")
    print("2. 🔄 Staging (balanced features)")
    print("3. 🚀 Production (comprehensive)")
    print("4. 🌟 Full (everything including UI)")
    print("5. 🛠️ Development tools only")
    print("6. 📋 Show requirements only")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        success = install_development()
    elif choice == "2":
        success = install_staging()
    elif choice == "3":
        success = install_production()
    elif choice == "4":
        success = install_full()
    elif choice == "5":
        success = install_dev_tools()
    elif choice == "6":
        show_requirements()
        return
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    if success:
        print("\n✅ Installation completed successfully!")
        print("\n🎯 Next steps:")
        print("   1. Test the installation: python -c 'import pyspark; print(\"PySpark installed!\")'")
        print("   2. Run examples: python environment_examples.py")
        print("   3. Check configuration: python -c 'from config_manager import ConfigManager; ConfigManager().print_config_summary()'")
    else:
        print("\n❌ Installation encountered errors")
        print("💡 Try installing manually: pip install -r requirements.txt")

def show_requirements():
    """Show requirements for different environments."""
    print("\n📋 REQUIREMENTS BY ENVIRONMENT")
    print("=" * 50)
    
    print("\n🔧 DEVELOPMENT (minimal):")
    print("   pyspark pandas numpy scikit-learn PyYAML joblib matplotlib")
    
    print("\n🔄 STAGING (balanced):")
    print("   All development packages plus:")
    print("   optuna xgboost seaborn openpyxl xlsxwriter")
    
    print("\n🚀 PRODUCTION (comprehensive):")
    print("   All staging packages plus:")
    print("   synapseml (optional, for LightGBM)")
    
    print("\n🌟 FULL (everything):")
    print("   All production packages plus:")
    print("   streamlit jupyter plotly")
    
    print("\n🛠️ DEVELOPMENT TOOLS:")
    print("   pytest black flake8 mypy isort")
    
    print("\n💾 Manual installation:")
    print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 