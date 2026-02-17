import os
import json
import random
import time
import base64
from datetime import datetime
from collections import deque
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import logging
import cv2

# Set up verbose logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_config():
    """Loads settings, prompts, and schema from config files."""
    try:
        with open('configs/settings.json', 'r') as f:
            settings = json.load(f)
        with open('configs/prompts.json', 'r') as f:
            prompts = json.load(f)
        with open('configs/schemas/video_response.schema.json', 'r') as f:
            schema = json.load(f)
        return settings, prompts, schema
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e.filename}")
        exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {e.doc.name}: {e.msg}")
        exit(1)

def get_video_files(input_dir, output_dir, max_items, shuffle):
    """Gets a list of video files to process."""
    all_videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
    processed_videos = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.json')}
    
    videos_to_process = [v for v in all_videos if os.path.splitext(v)[0] not in processed_videos]

    if shuffle:
        random.shuffle(videos_to_process)

    return videos_to_process[:max_items]

def calculate_cost(prompt_tokens, completion_tokens, model):
    """Calculate cost based on token usage and model pricing."""
    # OpenAI GPT-5 pricing (as of 2025) - verified from OpenAI pricing page
    pricing = {
        "gpt-5-mini": {
            "input": 0.25,   # $0.25 per million input tokens
            "output": 2.0    # $2.00 per million output tokens
        },
        "gpt-5-pro": {
            "input": 0.75,   # ESTIMATED - Please verify actual pricing from OpenAI pricing page
            "output": 5.0    # ESTIMATED - Please verify actual pricing from OpenAI pricing page
        },
        "gpt-5.1": {
            "input": 1.25,  # $1.25 per million input tokens
            "output": 10.0  # $10.00 per million output tokens
        },
        "gpt-5": {
            "input": 1.25,  # Same as 5.1
            "output": 10.0
        },
        "gpt-5-nano": {
            "input": 0.15,  # ESTIMATED - Please verify actual pricing from OpenAI pricing page
            "output": 1.0   # ESTIMATED - Please verify actual pricing from OpenAI pricing page
        }
    }
    
    # Determine which pricing to use based on model name
    model_key = None
    if "nano" in model.lower():
        model_key = "gpt-5-nano"
    elif "mini" in model.lower():
        model_key = "gpt-5-mini"
    elif "pro" in model.lower():
        model_key = "gpt-5-pro"
    elif "5.1" in model or "5-1" in model:
        model_key = "gpt-5.1"
    else:
        model_key = "gpt-5"  # Default to GPT-5/5.1 pricing
    
    if model_key not in pricing:
        # Fallback to GPT-5-mini pricing if model not recognized
        model_key = "gpt-5-mini"
        logging.warning(f"Unknown model {model}, using GPT-5-mini pricing")
    
    input_cost_per_million = pricing[model_key]["input"]
    output_cost_per_million = pricing[model_key]["output"]
    
    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
    output_cost = (completion_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost
    
    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "pricing_model": model_key
    }

def extract_frames(video_path, fps):
    """Extracts frames from video at specified fps and returns as base64 encoded images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = f"{video_width}x{video_height}"
    
    frame_interval = int(video_fps / fps) if fps > 0 else 1
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Encode frame as JPEG (smaller than PNG)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            extracted_frames.append(frame_base64)
        
        frame_count += 1
    
    cap.release()
    return extracted_frames, video_fps, frame_count, resolution

def process_video(video_path, client, prompts, schema, model, frame_sampling_fps, settings):
    """Processes a single video and returns the caption with metadata."""
    try:
        # Extract frames from video
        logging.info(f"Extracting frames from {os.path.basename(video_path)} at {frame_sampling_fps} fps...")
        frames, video_fps, total_frames, resolution = extract_frames(video_path, frame_sampling_fps)
        num_images = len(frames)
        logging.info(f"Extracted {num_images} frames from video (original: {total_frames} frames at {video_fps:.2f} fps, resolution: {resolution})")
        
        # Build content array with text prompt and all frames
        content = [{"type": "text", "text": prompts['video']['user']}]
        for frame_base64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_base64}"
                }
            })
        
        logging.info(f"Sending request to {model} with {num_images} images...")
        
        # Create chat completion with all frames
        # GPT-5 models only support default temperature (1), not 0
        create_params = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": prompts['video']['system']
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "video_analysis",
                    "strict": True,
                    "schema": schema
                }
            }
        }
        
        # Only set temperature if model supports it (GPT-4 models support temperature=0)
        if not model.startswith('gpt-5'):
            create_params["temperature"] = 0
        
        response = client.chat.completions.create(**create_params)
        
        # Extract usage information directly from API response
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0
        
        # Calculate cost based on actual API-reported tokens
        cost_info = calculate_cost(prompt_tokens, completion_tokens, model)
        
        # Calculate average tokens per image/frame (prompt_tokens includes all images + text)
        # This gives us the actual token cost per image as reported by the API
        avg_tokens_per_image = prompt_tokens / num_images if num_images > 0 else 0
        avg_tokens_per_frame = avg_tokens_per_image  # Same value, different naming for clarity
        
        # Log verbose information with actual API-reported values
        logging.info(f"API Response for {os.path.basename(video_path)}:")
        logging.info(f"  - Images sent: {num_images}")
        logging.info(f"  - Prompt tokens (from API): {prompt_tokens:,} (includes {num_images} images + text)")
        logging.info(f"  - Average tokens per image (from API): {avg_tokens_per_image:.2f}")
        logging.info(f"  - Completion tokens (from API): {completion_tokens:,}")
        logging.info(f"  - Total tokens (from API): {total_tokens:,}")
        logging.info(f"  - Context window usage: {total_tokens:,} / 400,000 ({total_tokens/400000*100:.2f}%)")
        logging.info(f"  - Estimated cost: ${cost_info['total_cost_usd']:.6f} (Input: ${cost_info['input_cost_usd']:.6f}, Output: ${cost_info['output_cost_usd']:.6f})")
        
        result = response.choices[0].message.content
        
        # Get file size
        file_size_bytes = os.path.getsize(video_path)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        
        # Calculate video length in seconds
        video_length_seconds = total_frames / video_fps if video_fps > 0 else 0
        video_length_formatted = f"{int(video_length_seconds // 60)}:{(int(video_length_seconds % 60)):02d}" if video_length_seconds > 0 else "0:00"
        
        # Get current timestamp
        analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create metadata objects matching the requested format
        usage_metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost_info['total_cost_usd'], 6)
        }
        
        file_metadata = {
            "filename": os.path.basename(video_path),
            "file_size_mb": file_size_mb,
            "analysis_timestamp": analysis_timestamp,
            "model_used": model,
            "frame_sampling_enabled": True,
            "frame_sampling_fps": float(frame_sampling_fps),
            "sampling_method": "frame_extraction",
            "resolution": resolution,  # Native video resolution
            "number_of_images": num_images,  # Number of frames extracted
            "video_length_seconds": round(video_length_seconds, 2),
            "video_length_formatted": video_length_formatted,  # MM:SS format
            "average_tokens_per_frame": round(avg_tokens_per_frame, 2)  # Average tokens per frame/image
        }
        
        # Create simplified metadata object
        metadata = {
            "usage_metadata": usage_metadata,
            "file_metadata": file_metadata
        }
        
        return result, metadata
    except Exception as e:
        logging.error(f"Error processing video {os.path.basename(video_path)}: {e}")
        return None, None

def main():
    """Main function to run the video captioning process."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file.")
        exit(1)

    client = OpenAI(api_key=api_key)
    settings, prompts, schema = load_config()
    
    vid_caption_settings = settings.get('vid_caption', {})
    common_settings = settings.get('common', {})
    safety_settings = settings.get('safety', {})
    
    input_dir = common_settings.get('input_root_dir', './videos')
    base_output_dir = common_settings.get('output_dir', './output')
    
    model = vid_caption_settings.get('model', 'gpt-5-mini')
    # Create model-specific output directory
    output_dir = os.path.join(base_output_dir, model)
    os.makedirs(output_dir, exist_ok=True)

    max_items = vid_caption_settings.get('max_items', 100)
    shuffle = vid_caption_settings.get('shuffle', True)
    concurrency = vid_caption_settings.get('concurrency', 2)
    
    request_delay = safety_settings.get('request_delay_seconds', 0.5)
    rate_limits = safety_settings.get('rate_limits', {})
    
    # Get model-specific limits
    model_limits = rate_limits.get(model, {})
    max_rpm = model_limits.get('rpm', safety_settings.get('max_rpm', 60))
    max_tpm = model_limits.get('tpm', 500000)  # Default fallback

    # Pass model-specific output directory to check for already processed videos
    video_files = get_video_files(input_dir, output_dir, max_items, shuffle)
    
    if not video_files:
        logging.info("No new videos to process.")
        return

    logging.info(f"Found {len(video_files)} new videos to process.")
    logging.info(f"Rate limits for {model}: RPM={max_rpm}, TPM={max_tpm:,}")

    error_count = 0
    consecutive_429 = 0
    completed_count = 0
    
    frame_sampling_fps = vid_caption_settings.get('frame_sampling_fps', 5)
    
    # RPM and TPM tracking: store timestamps and token counts (thread-safe)
    request_timestamps = deque()
    token_timestamps = deque()  # Store (timestamp, token_count) tuples
    rpm_lock = Lock()
    tpm_lock = Lock()
    
    def wait_for_rate_limits(estimated_tokens=0):
        """Wait if RPM or TPM limit is reached (thread-safe)."""
        # Check RPM limit
        wait_time_rpm = 0
        with rpm_lock:
            current_time = time.time()
            # Remove timestamps older than 1 minute
            while request_timestamps and current_time - request_timestamps[0] > 60:
                request_timestamps.popleft()
            
            # Check if we've reached the RPM limit
            if len(request_timestamps) >= max_rpm:
                oldest_timestamp = request_timestamps[0]
                wait_until = oldest_timestamp + 60
                wait_time_rpm = max(0, wait_until - current_time)
        
        # Check TPM limit
        wait_time_tpm = 0
        with tpm_lock:
            current_time = time.time()
            # Remove token entries older than 1 minute and calculate current TPM usage
            while token_timestamps and current_time - token_timestamps[0][0] > 60:
                token_timestamps.popleft()
            
            # Calculate tokens used in the last minute
            tokens_in_last_minute = sum(tokens for _, tokens in token_timestamps)
            
            # Check if adding this request would exceed TPM limit
            if tokens_in_last_minute + estimated_tokens > max_tpm:
                # Find when we can send this request
                if token_timestamps:
                    # Calculate when oldest tokens will expire
                    oldest_timestamp = token_timestamps[0][0]
                    wait_until = oldest_timestamp + 60
                    wait_time_tpm = max(0, wait_until - current_time)
                else:
                    # If no tokens in queue, we can proceed
                    wait_time_tpm = 0
        
        # Wait for the longer of the two wait times
        wait_time = max(wait_time_rpm, wait_time_tpm)
        if wait_time > 0:
            if wait_time_rpm > 0:
                logging.warning(f"RPM limit ({max_rpm}) reached. Waiting {wait_time_rpm:.2f} seconds...")
            if wait_time_tpm > 0:
                logging.warning(f"TPM limit ({max_tpm:,}) would be exceeded (current: {tokens_in_last_minute:,}, estimated: {estimated_tokens:,}). Waiting {wait_time_tpm:.2f} seconds...")
            time.sleep(wait_time)
            
            # Clean up old entries after waiting
            current_time = time.time()
            with rpm_lock:
                while request_timestamps and current_time - request_timestamps[0] > 60:
                    request_timestamps.popleft()
            with tpm_lock:
                while token_timestamps and current_time - token_timestamps[0][0] > 60:
                    token_timestamps.popleft()
        
        # Record this request timestamp
        with rpm_lock:
            request_timestamps.append(time.time())
    
    def record_token_usage(tokens):
        """Record token usage for TPM tracking."""
        with tpm_lock:
            token_timestamps.append((time.time(), tokens))
    
    def submit_with_rate_limit_check(video_path):
        """Submit a video for processing with rate limit checking."""
        # Quick frame count for token estimation
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / frame_sampling_fps) if frame_sampling_fps > 0 else 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        estimated_frames = total_frames // frame_interval if frame_interval > 0 else total_frames
        cap.release()
        
        # Estimate tokens (rough: ~1100 tokens per image + ~1000 for text)
        estimated_tokens = estimated_frames * 1100 + 1000
        
        wait_for_rate_limits(estimated_tokens)
        result, metadata = process_video(video_path, client, prompts, schema, model, frame_sampling_fps, settings)
        
        # Record actual token usage if available
        if metadata and 'usage_metadata' in metadata:
            total_tokens = metadata['usage_metadata'].get('total_tokens', 0)
            if total_tokens > 0:
                record_token_usage(total_tokens)
        
        return result, metadata
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(submit_with_rate_limit_check, os.path.join(input_dir, vf)): vf for vf in video_files}

        for future in as_completed(futures):
            completed_count += 1
            video_file = futures[future]
            video_name = os.path.splitext(video_file)[0]
            output_path = os.path.join(output_dir, f"{video_name}.json")
            
            try:
                result, metadata = future.result()
            except Exception as e:
                error_count += 1
                error_msg = str(e).lower()
                
                # Handle rate limits (429)
                if "rate limit" in error_msg or "429" in error_msg:
                    consecutive_429 += 1
                    consecutive_server_errors = 0
                # Handle server errors (500, 503, connection errors)
                elif "500" in error_msg or "503" in error_msg or "connection" in error_msg or "upstream connect" in error_msg:
                    logging.warning(f"Server error for {video_file}. Skipping this video and continuing after delay...")
                    time.sleep(request_delay * 10) # Wait longer (e.g., 5s if delay is 0.5s) to let server recover
                    continue # Skip this video immediately
                else:
                    consecutive_429 = 0
                
                logging.error(f"Exception processing video {video_file}: {e}")
                time.sleep(request_delay)
                continue

            if result:
                try:
                    caption_data = json.loads(result)
                    
                    # Add metadata to the output - usage_metadata and file_metadata at top level
                    output_data = {
                        **caption_data,
                        "usage_metadata": metadata.get("usage_metadata", {}),
                        "file_metadata": metadata.get("file_metadata", {})
                    }
                    
                    with open(output_path, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    logging.info(f"Successfully processed and saved captions for {video_file}")
                    consecutive_429 = 0
                    consecutive_server_errors = 0
                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON for {video_file}. Response was: {result}")
                    error_count += 1
                except Exception as e:
                    logging.error(f"An unexpected error occurred while saving the caption for {video_file}: {e}")
                    error_count += 1
            else:
                error_count += 1

            # Safety checks
            if safety_settings.get('enable_safety_monitor'):
                if safety_settings.get('max_consecutive_429') and consecutive_429 >= safety_settings['max_consecutive_429']:
                    logging.error("Exceeded maximum consecutive rate limit errors. Stopping.")
                    break
                
                if completed_count > 0 and safety_settings.get('max_error_rate') and (error_count / completed_count) > safety_settings['max_error_rate']:
                    logging.error("Exceeded maximum error rate. Stopping.")
                    break

            time.sleep(request_delay)

if __name__ == "__main__":
    main()
