# Installation

This guide will walk you through the process of installing the AMT package and its dependencies.

## Prerequisites

Before installing AMT, make sure you have the following prerequisites:

- Python 3.9 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation Methods

### Method 1: Install from GitHub (Recommended)

This is the recommended method as it gives you access to the latest code and allows you to make changes if needed.

```bash
# Clone the repository
git clone https://github.com/username/AMT.git
cd AMT

# Install the package in development mode
pip install -e .
```

This will install the package in "editable" mode, which means that changes to the source code will be immediately reflected without the need to reinstall.

### Method 2: Install from PyPI

```bash
# Install from PyPI
pip install amt-music
```

Note: The package name on PyPI may be different. Check the project documentation for the correct package name.

## Installing Development Dependencies

If you plan to contribute to the project or run tests, you should install the development dependencies:

```bash
# Install development dependencies
pip install -e ".[dev]"
```

This will install additional packages required for development, such as:

- pytest (for running tests)
- black, isort, ruff (for code formatting and linting)
- mypy (for type checking)
- mkdocs (for building documentation)

## Verifying the Installation

To verify that the installation was successful, run:

```bash
# Check the installed version
amt --version

# Show help information
amt --help
```

## Platform-Specific Notes

### Windows

On Windows, you may need to install additional dependencies for MIDI support:

```bash
# Install Windows-specific dependencies
pip install pypiwin32
```

### macOS

On macOS, you may need to install PortMIDI for MIDI support:

```bash
# Install PortMIDI using Homebrew
brew install portmidi

# Then install python-rtmidi
pip install python-rtmidi
```

### Linux

On Linux, you may need to install ALSA development libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libasound2-dev

# Fedora/RHEL/CentOS
sudo dnf install alsa-lib-devel

# Then install python-rtmidi
pip install python-rtmidi
```

## Troubleshooting

If you encounter any issues during installation, try the following:

1. Make sure you have the latest version of pip:
   ```bash
   pip install --upgrade pip
   ```

2. If you're having issues with dependencies, try installing them separately:
   ```bash
   pip install numpy torch pretty_midi mido tqdm
   ```

3. If you're still having issues, check the [GitHub Issues](https://github.com/username/AMT/issues) to see if others have encountered the same problem.

## Next Steps

Now that you have installed AMT, you can:

- Learn about [Configuration](configuration.md)
- Start [Collecting Data](data-collection.md)
- Explore the [API Reference](../api/index.md) 