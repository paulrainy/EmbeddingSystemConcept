#!/bin/bash

# Name of the virtual environment
VENV_NAME="venv"

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "Python is not installed."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
if [ "$PYTHON_VERSION" != "3.8.10" ]; then
    echo "Python version 3.8.10 is required. Installed version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv $VENV_NAME
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source $VENV_NAME/bin/activate

# Prompt the user to choose the requirements file to use
echo "Choose the requirements file to install:"
echo "1) requirements_cpu.txt"
echo "2) requirements_gpu.txt"
# shellcheck disable=SC2162
read -p "Enter the number (1 or 2): " CHOICE

if [ "$CHOICE" == "1" ]; then
    REQUIREMENTS_FILE="./requirements_cpu.txt"
elif [ "$CHOICE" == "2" ]; then
    REQUIREMENTS_FILE="./requirements_gpu.txt"
else
    echo "Invalid choice. Script is exiting."
    exit 1
fi

# Install the required packages
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing packages from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "File $REQUIREMENTS_FILE not found."
    exit 1
fi

echo "Starting milvus with docker-compose..."
sudo docker-compose up -f docker-compose.yml -d

# Проверяем состояние контейнера
while true; do
    # Получаем состояние контейнера
    state=$(sudo docker-compose ps --status "status=running" --filter "health=healthy" --format "{{.State}}" milvus-standalone)

    # Check if the state is "Up (healthy)"
    if [[ "$state" == "Up (healthy)" ]]; then
        echo "The container is up and running normally."
        break
    else
        echo "Waiting... Container state: $state"
        sleep 5
    fi
done

echo "Running the Python script..."
python main.py

