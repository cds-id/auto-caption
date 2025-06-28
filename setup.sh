#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Auto-Caption Setup Script ===${NC}"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}Found Python version: ${PYTHON_VERSION}${NC}"

# Check if Python version is 3.8 or higher
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.8 or higher is required. Current version: ${PYTHON_VERSION}${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing old one...${NC}"
    rm -rf venv
fi

python3 -m venv venv

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}Warning: ffmpeg is not installed.${NC}"
    echo -e "${YELLOW}Please install ffmpeg using one of the following commands:${NC}"
    echo -e "  - Ubuntu/Debian: ${GREEN}sudo apt-get install ffmpeg${NC}"
    echo -e "  - macOS: ${GREEN}brew install ffmpeg${NC}"
    echo -e "  - Fedora: ${GREEN}sudo dnf install ffmpeg${NC}"
    echo -e "  - Arch Linux: ${GREEN}sudo pacman -S ffmpeg${NC}"
else
    echo -e "${GREEN}ffmpeg is already installed.${NC}"
fi

# Install the package in development mode
echo -e "${YELLOW}Installing auto-caption in development mode...${NC}"
pip install -e .

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "${GREEN}To activate the virtual environment, run:${NC}"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "${GREEN}To use the CLI tool after activation:${NC}"
echo -e "  ${YELLOW}auto-caption --help${NC}"
echo ""
echo -e "${GREEN}To deactivate the virtual environment when done:${NC}"
echo -e "  ${YELLOW}deactivate${NC}"