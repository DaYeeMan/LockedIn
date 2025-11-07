#!/usr/bin/env python3
"""
PyTorch installation helper script.

This script helps install PyTorch with the appropriate configuration
for the SABR volatility surface modeling project.
"""

import subprocess
import sys
import platform


def detect_system():
    """Detect system configuration."""
    system = platform.system()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    print(f"System: {system}")
    print(f"Python: {python_version}")
    
    return system, python_version


def install_pytorch_cpu():
    """Install PyTorch CPU version."""
    print("\nInstalling PyTorch (CPU version)...")
    
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch CPU installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def install_pytorch_cuda():
    """Install PyTorch with CUDA support."""
    print("\nInstalling PyTorch (CUDA version)...")
    
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ PyTorch CUDA installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch CUDA installation failed: {e}")
        print(f"Error output: {e.stderr}")
        print("Falling back to CPU version...")
        return install_pytorch_cpu()


def test_pytorch_installation():
    """Test PyTorch installation."""
    try:
        import torch
        print(f"\n✅ PyTorch {torch.__version__} installed successfully!")
        
        # Test basic functionality
        x = torch.randn(2, 3)
        print(f"✅ Basic tensor operations work: {x.shape}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available (CPU-only mode)")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False


def main():
    """Main installation function."""
    print("PYTORCH INSTALLATION FOR SABR VOLATILITY SURFACE MODELING")
    print("=" * 60)
    
    # Detect system
    system, python_version = detect_system()
    
    # Check if PyTorch is already installed
    try:
        import torch
        print(f"\n✅ PyTorch {torch.__version__} is already installed!")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  Running in CPU-only mode")
        
        return 0
        
    except ImportError:
        print("\nPyTorch not found. Installing...")
    
    # Ask user for installation preference
    print("\nInstallation options:")
    print("1. CPU-only (recommended for most users)")
    print("2. CUDA (if you have NVIDIA GPU)")
    print("3. Auto-detect")
    
    choice = input("\nEnter your choice (1-3) [default: 1]: ").strip()
    
    if choice == "2":
        success = install_pytorch_cuda()
    elif choice == "3":
        # Try CUDA first, fall back to CPU
        print("Auto-detecting GPU support...")
        success = install_pytorch_cuda()
    else:
        # Default to CPU
        success = install_pytorch_cpu()
    
    if success:
        # Test installation
        success = test_pytorch_installation()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 PYTORCH INSTALLATION COMPLETE!")
        print("=" * 60)
        print("\nYou can now run the complete SABR pipeline:")
        print("  python check_pipeline.py")
        print("  python run_experiment.py")
        print("\nOr run individual components:")
        print("  python generate_training_data.py")
        print("  python main_training.py --data-dir data/processed")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ PYTORCH INSTALLATION FAILED")
        print("=" * 60)
        print("\nManual installation options:")
        print("1. CPU-only: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("2. CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Visit: https://pytorch.org/get-started/locally/")
        return 1


if __name__ == "__main__":
    sys.exit(main())