#!/bin/bash
set -e

echo "[INFO] Pulling latest images..."
docker compose pull

echo "[INFO] Restarting services..."
docker compose up -d --build

echo "[INFO] Removing unused images..."
docker image prune -f

echo "[INFO] Deployment complete!"
docker compose ps
