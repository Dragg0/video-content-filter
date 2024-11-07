import sys
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_env_variables():
    """Check if all required environment variables are set"""
    load_dotenv()
    
    env_vars = {
        'FLASK_APP': 'Flask application entry point',
        'UPLOAD_FOLDER': 'Directory for uploaded videos',
        'TORCH_DEVICE': 'PyTorch compute device (cuda/cpu)',
        'WHISPER_MODEL_SIZE': 'Whisper model size configuration',
        'DETECTION_CONFIDENCE': 'Confidence threshold for detection',
        'TRACKING_CONFIDENCE': 'Confidence threshold for tracking'
    }
    
    missing_vars = []
    for var, description in env_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        logger.warning("Missing environment variables:")
        for var in missing_vars:
            logger.warning(f"  - {var}")
        return False
    return True

def check_environment():
    """Verify all required components are properly installed"""
    results = {
        'success': True,
        'issues': []
    }
    
    # Check Python version
    logger.info(f"Python Version: {sys.version}")
    
    # Check PyTorch and CUDA
    try:
        import torch
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    except ImportError:
        results['success'] = False
        results['issues'].append("PyTorch not installed correctly")
    except Exception as e:
        results['success'] = False
        results['issues'].append(f"PyTorch error: {str(e)}")

    # Check OpenCV
    try:
        import cv2
        logger.info(f"OpenCV Version: {cv2.__version__}")
    except ImportError:
        results['success'] = False
        results['issues'].append("OpenCV not installed correctly")

    # Check MediaPipe
    try:
        import mediapipe as mp
        logger.info(f"MediaPipe Version: {mp.__version__}")
    except ImportError:
        results['success'] = False
        results['issues'].append("MediaPipe not installed correctly")

    # Check Transformers
    try:
        import transformers
        logger.info(f"Transformers Version: {transformers.__version__}")
    except ImportError:
        results['success'] = False
        results['issues'].append("Transformers not installed correctly")

    # Check Flask
    try:
        import flask
        logger.info(f"Flask Version: {flask.__version__}")
    except ImportError:
        results['success'] = False
        results['issues'].append("Flask not installed correctly")

    # Check Faster Whisper
    try:
        from faster_whisper import WhisperModel
        logger.info("Faster Whisper available")
    except ImportError:
        results['success'] = False
        results['issues'].append("Faster Whisper not installed correctly")

    # Check required directories
    required_dirs = ['uploads', 'models', '.cache']
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                results['success'] = False
                results['issues'].append(f"Failed to create {directory} directory: {str(e)}")
        else:
            logger.info(f"Directory exists: {directory}")

    return results

def main():
    logger.info("Starting environment check...")
    
    # Check environment variables
    logger.info("\n=== Checking Environment Variables ===")
    env_check = check_env_variables()
    
    # Check dependencies and system
    logger.info("\n=== Checking Dependencies and System ===")
    dep_check = check_environment()
    
    # Final results
    if env_check and dep_check['success']:
        logger.info("\n✅ All checks passed!")
    else:
        logger.error("\n❌ Some checks failed!")
        if not env_check:
            logger.error("Environment variables need to be configured")
        if dep_check['issues']:
            logger.error("Dependency issues found:")
            for issue in dep_check['issues']:
                logger.error(f"  - {issue}")
        sys.exit(1)

if __name__ == "__main__":
    main()