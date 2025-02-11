# macOS Setup Guide for Interactive Chat Avatar

## Prerequisites

1. Install Homebrew (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install Python and necessary dependencies:
```bash
# Install Python
brew install python

# Install OpenBLAS (required for scipy)
brew install openblas
```

## Virtual Environment Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Set environment variables for OpenBLAS:
```bash
export OPENBLAS=$(brew --prefix openblas)
export CFLAGS="-I$OPENBLAS/include"
export LDFLAGS="-L$OPENBLAS/lib"
```

## Dependency Installation

1. Upgrade pip:
```bash
pip install --upgrade pip
```

2. Install dependencies with special scipy configuration:
```bash
pip install scipy==1.13.1 \
  --config-settings=setup-args="-Dblas=blas" \
  --config-settings=setup-args="-Dlapack=lapack"

pip install -r requirements.txt
```

## Additional Troubleshooting

- If you encounter issues with torch, use the MPS (Metal Performance Shaders) version:
```bash
pip install torch torchvision torchaudio
```

- For LivePortrait, you might need additional setup for computer vision libraries.

## Common Issues

1. **OpenBLAS not found**: Ensure you've installed it via Homebrew and set the environment variables.
2. **Scipy installation fails**: Try the configuration steps above.
3. **Torch compatibility**: Ensure you're using the version compatible with Apple Silicon.

## Support

If you encounter persistent issues, please file an issue on the GitHub repository with detailed error logs.