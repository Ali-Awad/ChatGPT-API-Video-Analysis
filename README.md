# ChatGPT API Video Analysis

A Python application that uses the OpenAI GPT API to analyze dashcam and traffic video footage. It extracts frames from videos, sends them to GPT vision models, and returns structured JSON describing the scene, weather conditions, camera perspective, and any hazardous events.

## Features

- **Frame extraction** from video files at a configurable FPS using OpenCV
- **Concurrent processing** with configurable thread pool workers
- **Rate limit management** with automatic RPM/TPM tracking and backoff
- **Structured JSON output** enforced via JSON Schema (`response_format`)
- **Cost estimation** per request based on model-specific token pricing
- **Safety monitoring** to halt processing on excessive errors or rate limits
- **Resumable** &mdash; automatically skips videos that already have output files

## Project Structure

```
.
├── main.py                              # Main video analysis pipeline
├── configs/
│   ├── settings.json                    # Model, concurrency, rate limit settings
│   ├── prompts.json                     # System and user prompts
│   └── schemas/
│       └── video_response.schema.json   # JSON Schema for structured output
├── extra/
│   └── generate_video_report.py         # Utility to generate video metadata CSV
├── input/                               # Place video files here (git-ignored)
├── output/                              # Generated analysis JSON files (git-ignored)
├── requirements.txt
├── .env.example                         # Template for environment variables
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.8+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/ChatGPT-API-Video-Analysis.git
   cd ChatGPT-API-Video-Analysis
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key:**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and replace `your_api_key_here` with your actual OpenAI API key.

5. **Add video files:**

   Place your `.mp4`, `.mov`, or `.avi` video files in the `input/` directory.

## Usage

### Running the Analysis

```bash
python main.py
```

The script will:

1. Read configuration from `configs/settings.json`.
2. Scan `input/` for video files and skip any that already have output.
3. Extract frames at the configured FPS rate.
4. Send frames to the selected GPT model for analysis.
5. Save structured JSON results to `output/<model_name>/`.

### Generating a Video Report

To generate a CSV summary of video metadata (length, resolution, frame counts) without calling the API:

```bash
python extra/generate_video_report.py
```

## Configuration

All settings are in `configs/settings.json`:

| Section | Key | Description |
|---|---|---|
| `vid_caption` | `model` | GPT model to use (e.g., `gpt-5-mini`) |
| `vid_caption` | `frame_sampling_fps` | Frames to extract per second |
| `vid_caption` | `max_items` | Maximum videos to process per run |
| `vid_caption` | `concurrency` | Number of concurrent API requests |
| `vid_caption` | `shuffle` | Randomize video processing order |
| `safety` | `rate_limits` | Per-model RPM and TPM limits |
| `safety` | `max_error_rate` | Stop if error rate exceeds this threshold |
| `safety` | `request_delay_seconds` | Delay between requests |

### Supported Models

| Model | Input Cost | Output Cost | Notes |
|---|---|---|---|
| `gpt-5-nano` | ~$0.15/M tokens | ~$1.00/M tokens | Most cost-efficient |
| `gpt-5-mini` | $0.25/M tokens | $2.00/M tokens | Fast and cost-efficient |
| `gpt-5-pro` | ~$0.75/M tokens | ~$5.00/M tokens | Balanced |
| `gpt-5` / `gpt-5.1` | $1.25/M tokens | $10.00/M tokens | Highest quality |

*Prices marked with ~ are estimated. Verify on the [OpenAI pricing page](https://openai.com/pricing).*

## Output Format

Each analysis JSON file contains:

```json
{
  "scene description": "...",
  "weather": {
    "condition": "...",
    "winter weather": false
  },
  "camera view": "...",
  "hazardous event": {
    "present": true,
    "cause": "...",
    "effect": "...",
    "entities involved": ["..."]
  },
  "usage_metadata": {
    "prompt_tokens": 12345,
    "completion_tokens": 678,
    "total_tokens": 13023,
    "estimated_cost_usd": 0.003
  },
  "file_metadata": {
    "filename": "video.mp4",
    "file_size_mb": 12.5,
    "analysis_timestamp": "2025-12-01 15:30:00",
    "model_used": "gpt-5-mini",
    "resolution": "1920x1080",
    "number_of_images": 30,
    "video_length_seconds": 10.0
  }
}
```

## License

This project is provided as-is for research and educational purposes.
