# Auto-Caption with Jupyter in Docker

This guide explains how to run Auto-Caption in a Jupyter notebook environment using Docker.

## Quick Start

### 1. Basic Usage (CPU)

```bash
# Start Jupyter with auto-caption
./start-jupyter.sh

# Access Jupyter at http://localhost:8888
```

### 2. GPU Usage

```bash
# Start with GPU support
./start-jupyter.sh --gpu
```

## Manual Docker Commands

### Using Docker Compose

```bash
# Start with CPU
docker-compose up

# Start with GPU
docker-compose -f docker-compose.gpu.yml up

# Run in background
docker-compose up -d

# Stop containers
docker-compose down
```

### Build and Run

```bash
# Build the image
docker-compose build

# Build with GPU support
docker-compose -f docker-compose.gpu.yml build

# Force rebuild
docker-compose build --no-cache
```

## Directory Structure

After starting, you'll have these directories:

```
auto-caption/
├── notebooks/      # Your Jupyter notebooks
├── data/          # Input videos and data
├── outputs/       # Generated captions and videos
└── cache/         # Model cache (persistent)
```

## Using Auto-Caption in Jupyter

### Basic Example

```python
from auto_caption import CaptionGenerator, EmotionDetector

# Generate captions
generator = CaptionGenerator(model_name="base")
result = generator.generate_captions("data/video.mp4")

# Detect emotions
detector = EmotionDetector()
emotions = detector.process_video("data/video.mp4")
```

### Quick Processing

```python
from auto_caption.examples.jupyter_examples.jupyter_usage import quick_process

# Process video with one command
results = quick_process("data/my_video.mp4")
```

### Interactive Widget

```python
from auto_caption.examples.jupyter_examples.jupyter_usage import demo_emotion_styling

# Create interactive emotion styling widget
widget = demo_emotion_styling()
display(widget)
```

## Environment Variables

Create a `.env` file to customize settings:

```bash
# Copy template
cp .env.example .env

# Edit settings
nano .env
```

Key settings:
- `JUPYTER_PORT`: Change Jupyter port (default: 8888)
- `CUDA_VISIBLE_DEVICES`: Select GPU (default: 0)
- `WHISPER_MODEL`: Model size (tiny/base/small/medium/large)
- `DEFAULT_PLATFORM`: Target platform (GENERAL/TIKTOK/INSTAGRAM/YOUTUBE_SHORTS)

## GPU Setup

### Prerequisites

1. NVIDIA GPU with CUDA support
2. NVIDIA Docker runtime installed:
   ```bash
   # Install NVIDIA Docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### Verify GPU Access

In Jupyter, run:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
```

## Troubleshooting

### Port Already in Use

```bash
# Change port in .env file
JUPYTER_PORT=8889

# Or kill existing process
sudo lsof -ti:8888 | xargs kill -9
```

### Permission Issues

```bash
# Fix permissions
sudo chown -R $USER:$USER notebooks data outputs cache
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.7.0-base-ubuntu20.04 nvidia-smi
```

### Out of Memory

```bash
# Reduce batch size in code
generator = CaptionGenerator(
    model_name="tiny",  # Use smaller model
    device="cuda"
)

# Or limit GPU memory in Docker
docker run --gpus '"device=0"' --shm-size=8g ...
```

## Tips and Best Practices

### 1. Model Selection

- **tiny**: Fast, less accurate (39M parameters)
- **base**: Good balance (74M parameters)
- **small**: Better accuracy (244M parameters)
- **medium**: High accuracy (769M parameters)
- **large**: Best accuracy (1550M parameters)

### 2. Video Processing

```python
# Process large videos in chunks
generator = CaptionGenerator(
    model_name="base",
    chunk_length=30  # Process 30-second chunks
)
```

### 3. Batch Processing

```python
# Process multiple videos
import glob

video_files = glob.glob("data/*.mp4")
for video in video_files:
    result = quick_process(video)
    # Save results
    with open(f"outputs/{Path(video).stem}_captions.json", "w") as f:
        json.dump(result, f)
```

### 4. Memory Management

```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Delete large objects
del generator
del detector
```

## Advanced Usage

### Custom Docker Image

```dockerfile
# Extend the base image
FROM auto-caption-jupyter:latest

# Add your dependencies
RUN pip install your-package

# Add custom notebooks
COPY my_notebooks/ /workspace/notebooks/
```

### Running CLI Commands

```bash
# Run CLI in container
docker-compose run --rm jupyter-auto-caption \
    auto-caption generate /workspace/data/video.mp4 \
    --emotion --format ass

# Batch process
docker-compose run --rm jupyter-auto-caption \
    auto-caption batch /workspace/data \
    --emotion --platform tiktok
```

### Persistent Model Cache

Models are cached in the `cache/` directory. To pre-download:

```bash
# Download models
docker-compose run --rm jupyter-auto-caption python -c "
from auto_caption import CaptionGenerator, EmotionDetector
CaptionGenerator(model_name='base')
EmotionDetector()
print('Models downloaded!')
"
```

## Security Notes

The default configuration has no authentication for ease of use. For production:

1. Set a password in `.env`:
   ```bash
   JUPYTER_PASSWORD=your_secure_password
   ```

2. Use token authentication:
   ```bash
   JUPYTER_TOKEN=your_secure_token
   ```

3. Restrict network access:
   ```yaml
   # In docker-compose.yml
   ports:
     - "127.0.0.1:8888:8888"  # Local access only
   ```

## Stopping and Cleanup

```bash
# Stop containers
docker-compose down

# Remove containers and networks
docker-compose down -v

# Remove all data (careful!)
docker-compose down -v
rm -rf notebooks data outputs cache
```

## Further Resources

- [Example Notebooks](examples/jupyter_examples/)
- [API Documentation](docs/api.md)
- [CLI Documentation](docs/cli.md)
- [Docker Documentation](https://docs.docker.com/)
- [Jupyter Documentation](https://jupyter.org/documentation)