#!/bin/bash

# this gets the container id and assigns it as the node_id
export NODE_ID=$(hostname | grep -oE '[0-9]+$')

echo "Starting node with ID: $NODE_ID"

# Start your original command (e.g., your Python backend)
exec python backend.py