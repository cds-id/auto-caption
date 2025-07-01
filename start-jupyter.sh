#!/bin/bash

# Auto-Caption Jupyter Notebook Launcher
# This script starts a Jupyter Lab environment with auto-caption installed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_color "$BLUE" "🔍 Checking prerequisites..."

if ! command_exists docker; then
    print_color "$RED" "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    print_color "$RED" "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_color "$GREEN" "✅ Prerequisites satisfied"

# Create necessary directories
print_color "$BLUE" "📁 Creating directories..."
mkdir -p notebooks data cache outputs

# Check for GPU support
GPU_AVAILABLE=false
if command_exists nvidia-smi; then
    if nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        print_color "$GREEN" "🎮 GPU detected!"
    fi
fi

# Parse command line arguments
USE_GPU=false
BUILD_FRESH=false
DETACHED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            if [ "$GPU_AVAILABLE" = true ]; then
                USE_GPU=true
                print_color "$GREEN" "🚀 GPU mode enabled"
            else
                print_color "$YELLOW" "⚠️  GPU requested but not available, falling back to CPU"
            fi
            shift
            ;;
        --build)
            BUILD_FRESH=true
            print_color "$BLUE" "🔨 Fresh build requested"
            shift
            ;;
        --detached|-d)
            DETACHED=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu         Use GPU-enabled container (requires NVIDIA GPU)"
            echo "  --build       Force rebuild of Docker image"
            echo "  --detached    Run in detached mode"
            echo "  --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Start with CPU"
            echo "  $0 --gpu              # Start with GPU support"
            echo "  $0 --gpu --build      # Rebuild and start with GPU"
            exit 0
            ;;
        *)
            print_color "$RED" "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Select appropriate docker-compose file
if [ "$USE_GPU" = true ]; then
    COMPOSE_FILE="docker-compose.gpu.yml"
    DOCKERFILE="Dockerfile.jupyter.gpu"
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_color "$RED" "❌ GPU compose file not found: $COMPOSE_FILE"
        exit 1
    fi
else
    COMPOSE_FILE="docker-compose.yml"
    DOCKERFILE="Dockerfile.jupyter"
fi

print_color "$BLUE" "📋 Using compose file: $COMPOSE_FILE"

# Build if requested or if image doesn't exist
if [ "$BUILD_FRESH" = true ]; then
    print_color "$BLUE" "🔨 Building Docker image..."
    docker-compose -f "$COMPOSE_FILE" build
elif ! docker images | grep -q "auto-caption-jupyter"; then
    print_color "$YELLOW" "🔨 Image not found, building..."
    docker-compose -f "$COMPOSE_FILE" build
fi

# Stop any existing containers
print_color "$BLUE" "🛑 Stopping existing containers..."
docker-compose -f "$COMPOSE_FILE" down

# Start the container
print_color "$BLUE" "🚀 Starting Jupyter Lab..."
if [ "$DETACHED" = true ]; then
    docker-compose -f "$COMPOSE_FILE" up -d
    print_color "$GREEN" "✅ Container started in detached mode"
else
    # Start in foreground with proper signal handling
    trap 'docker-compose -f "$COMPOSE_FILE" down' INT TERM
    
    print_color "$GREEN" "✅ Starting Jupyter Lab..."
    echo ""
    print_color "$YELLOW" "📌 Access Jupyter Lab at: http://localhost:8888"
    print_color "$YELLOW" "📌 No password required"
    print_color "$YELLOW" "📌 Press Ctrl+C to stop"
    echo ""
    
    if [ "$USE_GPU" = true ]; then
        print_color "$GREEN" "🎮 GPU acceleration is enabled"
    else
        print_color "$BLUE" "💻 Running in CPU mode (use --gpu for GPU support)"
    fi
    echo ""
    
    docker-compose -f "$COMPOSE_FILE" up
fi

# Show logs if running in detached mode
if [ "$DETACHED" = true ]; then
    echo ""
    print_color "$GREEN" "✅ Jupyter Lab is starting..."
    print_color "$YELLOW" "📌 Access at: http://localhost:8888"
    print_color "$YELLOW" "📌 View logs: docker-compose -f $COMPOSE_FILE logs -f"
    print_color "$YELLOW" "📌 Stop: docker-compose -f $COMPOSE_FILE down"
    echo ""
    
    # Wait a moment and check if container is running
    sleep 3
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        print_color "$GREEN" "✅ Container is running successfully!"
        
        # Show GPU info if available
        if [ "$USE_GPU" = true ]; then
            echo ""
            print_color "$BLUE" "🎮 GPU Information:"
            docker-compose -f "$COMPOSE_FILE" exec -T jupyter-auto-caption-gpu python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
" 2>/dev/null || print_color "$YELLOW" "⚠️  Could not get GPU info"
        fi
    else
        print_color "$RED" "❌ Container failed to start. Check logs with:"
        echo "docker-compose -f $COMPOSE_FILE logs"
    fi
fi