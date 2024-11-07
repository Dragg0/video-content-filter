from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import cv2
import numpy as np
from src.analysis.nsfw_detector import NSFWDetector
import logging
import threading
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    UPLOAD_FOLDER = Path('uploads')
    PROCESSED_FOLDER = Path('processed_videos')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB limit

    # Create directories if they don't exist
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    PROCESSED_FOLDER.mkdir(exist_ok=True)

app.config.from_object(Config)

# Initialize NSFW detector
detector = NSFWDetector()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

class VideoProcessor:
    def __init__(self):
        self.processing_status = {}
    
    def blur_video(self, input_path: str, output_path: str, nsfw_timestamps: list):
        """Process video and blur NSFW segments"""
        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_number = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_number / fps
                
                # Check if current frame is in NSFW segment
                should_blur = any(abs(timestamp - current_time) < 1.0 
                                for timestamp in nsfw_timestamps)
                
                if should_blur:
                    # Apply heavy blurring
                    blurred = cv2.GaussianBlur(frame, (99, 99), 30)
                    out.write(blurred)
                else:
                    out.write(frame)
                
                frame_number += 1
                
                # Update processing status
                progress = (frame_number * 100) / cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.processing_status[output_path] = {
                    'progress': progress,
                    'status': 'processing'
                }
            
            cap.release()
            out.release()
            
            self.processing_status[output_path] = {
                'progress': 100,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            self.processing_status[output_path] = {
                'progress': 0,
                'status': 'error',
                'error': str(e)
            }

video_processor = VideoProcessor()

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Allowed file types: {", ".join(Config.ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        input_path = app.config['UPLOAD_FOLDER'] / safe_filename
        file.save(str(input_path))
        
        # Start processing in background
        output_filename = f"processed_{safe_filename}"
        output_path = app.config['PROCESSED_FOLDER'] / output_filename
        
        # Start NSFW detection and processing
        def process_video():
            try:
                # Detect NSFW content
                results = detector.analyze_video(str(input_path))
                
                # Blur detected segments
                video_processor.blur_video(
                    str(input_path),
                    str(output_path),
                    results['unsafe_timestamps']
                )
            except Exception as e:
                logger.error(f"Error in video processing: {str(e)}")
        
        thread = threading.Thread(target=process_video)
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded and processing started',
            'filename': safe_filename
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<filename>')
def get_status(filename):
    """Get processing status for a video"""
    output_path = str(app.config['PROCESSED_FOLDER'] / f"processed_{filename}")
    status = video_processor.processing_status.get(output_path, {
        'status': 'not_found',
        'progress': 0
    })
    return jsonify(status)

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed video"""
    try:
        return send_file(
            app.config['PROCESSED_FOLDER'] / filename,
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)