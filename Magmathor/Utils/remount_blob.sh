#!/bin/bash
# Remount Azure Blob Storage using blobfuse2

# Configuration
MOUNT_POINT="/mnt/magmathor"
CACHE_DIR="/tmp/blobfuse2_cache"
CONFIG_FILE="$HOME/blobfuse_magmathor.yaml"

# Unmount if already mounted (handle stale/hung mounts)
echo "Cleaning up any existing mount..."

# Kill any stuck blobfuse2 processes first
if pgrep -f "blobfuse2.*$MOUNT_POINT" > /dev/null; then
    echo "Killing stuck blobfuse2 processes..."
    sudo pkill -9 -f "blobfuse2.*$MOUNT_POINT" 2>/dev/null || true
    sleep 1
fi

# Try normal unmount first
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
    echo "Unmounting $MOUNT_POINT..."
    sudo fusermount -u "$MOUNT_POINT" 2>/dev/null || true
    sleep 1
fi

# If still mounted or stale, force lazy unmount
if mountpoint -q "$MOUNT_POINT" 2>/dev/null || [ -d "$MOUNT_POINT" ]; then
    echo "Force unmounting (lazy)..."
    sudo umount -l "$MOUNT_POINT" 2>/dev/null || true
    sudo fusermount -uz "$MOUNT_POINT" 2>/dev/null || true
    sleep 2
fi

# Ensure mount point exists and is accessible
echo "Setting up mount point..."
sudo mkdir -p "$MOUNT_POINT" 2>/dev/null || true
sudo chown $USER:$USER "$MOUNT_POINT" 2>/dev/null || true

# Ensure cache directory exists and is empty
echo "Clearing cache directory..."
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

# Mount using blobfuse2
echo "Mounting blob storage to $MOUNT_POINT..."
blobfuse2 mount "$MOUNT_POINT" --config-file="$CONFIG_FILE"

# Verify mount
if mountpoint -q "$MOUNT_POINT"; then
    echo "Successfully mounted blob storage at $MOUNT_POINT"
    ls -la "$MOUNT_POINT"
else
    echo "Failed to mount blob storage"
    exit 1
fi
