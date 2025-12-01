#!/bin/bash

set -e  # Exit on error
set -o pipefail


# -------- Rapidsnark dependencies -------
echo "📦 Installing Rapidsnark dependencies"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install build-essential cmake libgmp-dev libsodium-dev nasm curl m4
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install cmake gmp libsodium nasm wget
fi

# -------- Python Setup --------
echo "🔧 Setting up Python environment..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.10 or 3.11 manually."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" ]]; then
    echo "❌ Python version $PYTHON_VERSION found. Please use Python 3.10 or 3.11."
    exit 1
fi

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "❌ Please activate your virtual environment before running this script."
    exit 1
fi

pip3 install -U setuptools

pip3 install --upgrade pip
pip3 install flake8 pytest

if [ -f requirements.txt ]; then
    echo "📦 Installing Python dependencies from requirements.txt..."
    pip3 install -r requirements.txt
fi

# -------- NodeJS + Circom + SnarkJS --------
echo "🔧 Setting up Node.js, Circom and SnarkJS..."

if ! command -v node &> /dev/null; then
    echo "📦 Installing Node.js 16.x..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install node@18
        brew link --force --overwrite node@18
    fi
fi

echo "📦 Installing NPM packages..."
npm install
npm install -g snarkjs

if ! command -v circom &> /usr/local/bin/circom; then
    echo "📦 Installing Circom..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CIRCOM_PREBUILD_OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        CIRCOM_PREBUILD_OS="macos"
    fi

    sudo wget -O /usr/local/bin/circom "https://github.com/iden3/circom/releases/download/v2.1.7/circom-$CIRCOM_PREBUILD_OS-amd64"

    sudo chmod +x /usr/local/bin/circom
fi

# -------- Rust Setup --------
echo "🔧 Setting up Rust toolchain..."

if ! command -v rustup &> /dev/null; then
    echo "📦 Installing Rust..."
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source $HOME/.cargo/env
fi

echo "📦 Installing Rust toolchain 1.90.0..."
rustup install 1.90.0
rustup default 1.90.0
rustup component add rustfmt clippy


# -------- Building UltraGroth -------
echo "🔧 Building ultragroth..."
pushd "tf_bionetta/repos/ultragroth" > /dev/null

git submodule init
git submodule update

SYSTEM=$(uname -s)-$(uname -m)
echo "Detected system: $SYSTEM"

case "$SYSTEM" in
    Darwin-x86_64|Linux-x86_64)
        ./build_gmp.sh host
        make host
        ;;
    Darwin-arm64)
        ./build_gmp.sh macos_arm64
        make macos_arm64
        ;;
    Linux-aarch64)
        ./build_gmp.sh host
        make host_arm64
        ;;
    *)
        echo "⚠️ Unsupported system: $SYSTEM"
        ;;
esac

popd > /dev/null
echo "✅ ultragroth built successfully"

# -------- Done --------
echo "✅ All dependencies installed successfully."
