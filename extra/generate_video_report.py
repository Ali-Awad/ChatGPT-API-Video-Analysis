import os
import json
import cv2
import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_settings():
    """Loads settings from config file."""
    try:
        with open('configs/settings.json', 'r') as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        logging.error("configs/settings.json not found.")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return {}

def get_video_files(input_dir):
    """Gets a list of video files recursively."""
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi')):
                video_files.append(os.path.join(root, file))
    return video_files

def analyze_video(video_path, fps=5):
    """Analyzes a video file to get length, resolution, and extracted frames info."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return None

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        resolution = f"{video_width}x{video_height}"
        video_length_seconds = total_frames / video_fps if video_fps > 0 else 0
        
        frame_interval = int(video_fps / fps) if fps > 0 else 1
        
        extracted_count = 0
        total_size_bytes = 0
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Encode frame as JPEG to get size (quality 85 as in main.py)
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    extracted_count += 1
                    total_size_bytes += len(buffer)
            
            frame_idx += 1

        cap.release()
        
        return {
            "filename": os.path.basename(video_path),
            "path": video_path,
            "length_seconds": round(video_length_seconds, 2),
            "resolution": resolution,
            "extracted_frames_count": extracted_count,
            "extracted_frames_size_mb": round(total_size_bytes / (1024 * 1024), 2)
        }
    except Exception as e:
        logging.error(f"Error analyzing {video_path}: {e}")
        return None

def main():
    settings = load_settings()
    common_settings = settings.get('common', {})
    input_dir = common_settings.get('input_root_dir', './videos')
    output_csv = 'video_analysis_report.csv'
    
    # Use a reasonable default for local processing, ignoring the API-focused setting if it's low
    # But allowing it to be higher if set higher.
    cpu_count = os.cpu_count() or 4
    max_workers = max(cpu_count, 4)
    
    if not os.path.exists(input_dir):
        logging.error(f"Input directory not found: {input_dir}")
        return

    video_files = get_video_files(input_dir)
    total_videos = len(video_files)
    logging.info(f"Found {total_videos} video files in {input_dir}. Processing with {max_workers} workers...")
    
    if not video_files:
        return

    results = []
    completed_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(analyze_video, video, fps=5): video for video in video_files}
        
        for future in as_completed(future_to_video):
            completed_count += 1
            data = future.result()
            if data:
                results.append(data)
            
            # Progress update
            elapsed_time = time.time() - start_time
            avg_time_per_video = elapsed_time / completed_count
            remaining_videos = total_videos - completed_count
            estimated_remaining = remaining_videos * avg_time_per_video
            
            print(f"Progress: {completed_count}/{total_videos} ({completed_count/total_videos*100:.1f}%) - "
                  f"ETA: {estimated_remaining:.0f}s - "
                  f"Processed {os.path.basename(future_to_video[future])}", end='\r')

    print("\nProcessing complete. Writing report...")

    # Write results to CSV
    if results:
        # Sort by filename for consistent output
        results.sort(key=lambda x: x['filename'])
        
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'path', 'length_seconds', 'resolution', 'extracted_frames_count', 'extracted_frames_size_mb']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(results)
            
        logging.info(f"Report saved to {output_csv} with {len(results)} rows.")
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    main()
