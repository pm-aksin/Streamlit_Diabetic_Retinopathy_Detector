import subprocess
import sys

# List of packages to check and install
required_packages = [
    'streamlit', 'torch', 'torchvision', 'Pillow', 'timm', 'reportlab', 'sqlite3', 'datetime'
]

# Function to install missing packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install missing packages
for package in required_packages:
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} not found. Installing...")
        install_package(package)
