import os
import sys
import subprocess
import platform

def create_venv():
    """Create virtual environment and install requirements"""
    print("🚀 Starting project setup...")
    
    # Determine the python executable to use
    python_cmd = 'python' if platform.system() == 'Windows' else 'python3'
    
    try:
        # Create virtual environment
        print("📦 Creating virtual environment...")
        subprocess.run([python_cmd, '-m', 'venv', 'venv'], check=True)
        
        # Determine the pip executable path
        if platform.system() == 'Windows':
            pip_path = os.path.join('venv', 'Scripts', 'pip')
            activate_script = os.path.join('venv', 'Scripts', 'activate')
        else:
            pip_path = os.path.join('venv', 'bin', 'pip')
            activate_script = os.path.join('venv', 'bin', 'activate')
        
        # Upgrade pip
        print("⬆️ Upgrading pip...")
        subprocess.run([pip_path, 'install', '--upgrade', 'pip'], check=True)
        
        # Install requirements
        print("📥 Installing requirements...")
        subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
        
        print("\n✅ Setup completed successfully!")
        print(f"\nTo activate the virtual environment:")
        if platform.system() == 'Windows':
            print(f"    Run: .\\venv\\Scripts\\activate")
        else:
            print(f"    Run: source venv/bin/activate")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during setup: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    create_venv()