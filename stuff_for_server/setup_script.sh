pip install -r requirements.txt --break-system-packages

# Add /home/user/.local/bin to PATH if not already present
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "Added /home/user/.local/bin to PATH in ~/.bashrc"
else
    echo "/home/user/.local/bin is already in PATH"
fi

# Reload the shell configuration
source ~/.bashrc
echo "PATH updated. Current PATH: $PATH"