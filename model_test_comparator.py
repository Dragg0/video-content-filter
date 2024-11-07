import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from ultralytics import YOLO
import deepai
from PIL import Image
import cv2
import os
import json
from pathlib import Path
import numpy as np

# Set DeepAI API Key from environment variable
deepai_api_key = os.getenv("DEEPAI_API_KEY")
if not deepai_api_key:
    raise ValueError("Please set the DEEPAI_API_KEY environment variable.")
deepai.set_api_key(deepai_api_key)

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
yolo_model = YOLO("yolov8m-seg.pt")

# Define frame folder and output file
frame_folder = "detected_frames"
output_file = "model_test_results.jsonl"

# Get frames from 25 to 30 seconds
frames_to_test = sorted(Path(frame_folder).glob("frame_*.jpg"))
test_frames = [f for f in frames_to_test if 25 <= int(f.stem.split('_')[-1]) / 30 < 30]

def analyze_blip(image: Image) -> str:
    """Get descriptive output from BLIP model."""
    print("Running BLIP analysis...")
    inputs = blip_processor(images=image, text="Describe in detail, noting any partially exposed body parts", return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs, max_new_tokens=50)
    description = blip_processor.decode(outputs[0], skip_special_tokens=True)
    print("BLIP description:", description)
    return description

def analyze_clip(image: Image) -> str:
    """Get descriptive output from CLIP model."""
    print("Running CLIP analysis...")
    inputs = clip_processor(text=["woman holding soccer ball, partial buttocks exposed"], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    description = "CLIP output with context prompt"
    print("CLIP description:", description)
    return description

def analyze_yolo(image: np.ndarray) -> str:
    """Get object detection output from YOLO model."""
    print("Running YOLO analysis...")
    result = yolo_model(image)
    yolo_result = result.pandas().xyxy[0].to_json(orient="records")
    print("YOLO results:", yolo_result)
    return yolo_result

def analyze_nsfw(image_path: str) -> str:
    """Get explicit content analysis from NSFW model."""
    print("Running NSFW analysis...")
    try:
        result = deepai.ImageRecognition.create(image=open(image_path, 'rb'))
        nsfw_result = result['output']
        print("NSFW result:", nsfw_result)
        return nsfw_result
    except Exception as e:
        print("NSFW model error:", e)
        return "NSFW model error"

def run_test():
    results = []

    for frame_path in test_frames:
        print(f"\nProcessing {frame_path}...")

        image = Image.open(frame_path)
        cv_image = cv2.imread(str(frame_path))

        # BLIP Analysis
        blip_desc = analyze_blip(image)

        # CLIP Analysis
        clip_desc = analyze_clip(image)

        # YOLO Detection
        yolo_desc = analyze_yolo(cv_image)

        # NSFW Analysis
        nsfw_desc = analyze_nsfw(str(frame_path))

        # Collect results
        result = {
            "frame": str(frame_path),
            "BLIP": blip_desc,
            "CLIP": clip_desc,
            "YOLO": yolo_desc,
            "NSFW": nsfw_desc
        }
        results.append(result)

        # Save to output file incrementally
        print("Saving result to file...")
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    print(f"All results saved to {output_file}")

if __name__ == "__main__":
    run_test()
