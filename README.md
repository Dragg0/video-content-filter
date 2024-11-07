# Personal Video Content Filter

## Project Overview
A personal-use application for intelligent content filtering of video media. This tool helps users customize their viewing experience by allowing them to skip or mute specific types of content based on their preferences.

## Goals
- Create an AI-powered system to automatically detect and categorize content in videos
- Provide customizable filtering options for:
  - Visual content
  - Audio content
  - Language/dialogue
  - Scene-level content

## Technical Features
- **Content Detection**
  - Computer vision for visual content analysis
  - Audio processing for speech recognition
  - Scene classification
  - Real-time content analysis

- **Filtering Capabilities**
  - Timestamp-based content skipping
  - Audio muting options
  - Scene filtering
  - Customizable filter categories

- **User Interface**
  - Upload interface for video files
  - Filter preference settings
  - Preview and adjustment capabilities
  - Export filtered video

## Technical Stack
- Python
- PyTorch for ML models
- OpenCV for video processing
- Faster Whisper for speech recognition
- MediaPipe for additional visual analysis
- Flask for web interface
- Transformers for advanced ML tasks

## Development Phases
1. **Phase 1: Core Infrastructure**
   - Basic video processing pipeline
   - File upload and handling
   - Database setup for timestamps

2. **Phase 2: AI Integration**
   - Implement visual content detection
   - Add speech recognition
   - Develop scene analysis

3. **Phase 3: User Interface**
   - Create web interface
   - Add filter customization
   - Implement video preview

4. **Phase 4: Optimization**
   - Improve processing speed
   - Enhance detection accuracy
   - Add batch processing capabilities

## Note on Usage
This project is intended for personal use only, allowing individuals to filter their own legally owned content based on their preferences.

# Personal Video Content Filter

[Previous sections remain the same until Current Status]

## Project Setup & Progress

### ✅ Completed
- Initial project structure created
- Basic dependency setup with requirements.txt:
  ```
  torch
  transformers
  Pillow
  opencv-python
  ultralytics
  deepai
  numpy
  mediapipe
  flask
  faster-whisper
  python-dotenv
  ```
- Environment checking system (check_env.py)
  - Validates all required dependencies
  - Checks for CUDA availability
  - Verifies environment variables
  - Creates necessary directories (uploads, models, .cache)

### 🚧 In Progress
- Setting up environment variables (.env configuration)
- Installing remaining dependencies identified by check_env.py:
  - MediaPipe
  - Flask
  - Faster Whisper

### 📁 Current Project Structure
```
.
│   analyze_frame_enhanced.py
│   app.py
│   check_env.py
│   model_test_comparator.py
│   requirements.txt
│   Vidangel Clone.code-workspace
│
├───.vscode
│       launch.json
│       settings.json
│
├───templates
│       upload.html
│
├───uploads
├───models
└───.cache
```

### 🔜 Next Steps
1. Complete dependency installation
2. Configure .env file
3. Set up basic Flask application structure
4. Begin implementing video upload functionality

## Current Status
In development - Completing initial environment setup and resolving dependency issues.