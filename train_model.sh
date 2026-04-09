#!/usr/bin/env bash
# CrowdGuard - Full Model Training Pipeline
# Runs all steps in order. Exits on first failure.
set -e

echo "============================================"
echo " CrowdGuard Model Training Pipeline"
echo "============================================"

echo ""
echo "[1/5] Preparing VisDrone dataset..."
python backend/ml/dataset_prep.py
echo "      Done."

echo ""
echo "[2/5] Evaluating pretrained baseline..."
python backend/ml/evaluate_baseline.py
echo "      Done."

echo ""
echo "[3/5] Fine-tuning YOLOv8n on VisDrone..."
python backend/ml/train.py
echo "      Done."

echo ""
echo "[4/5] Comparing models..."
python backend/ml/compare_models.py
echo "      Done."

echo ""
echo "[5/5] Generating metric plots..."
python backend/ml/plot_metrics.py
echo "      Done."

echo ""
echo "============================================"
echo " Pipeline complete."
echo " Weights : weights/crowdguard_visdrone.pt"
echo " Metrics : results/"
echo " Plots   : results/plots/"
echo "============================================"
