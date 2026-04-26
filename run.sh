#!/bin/bash

# Music Edit Learning Training and Evaluation Script
# This script trains and evaluates baseline and edit-aware models with different alpha values

echo "Starting Music Edit Learning Training Pipeline"
echo "=============================================="

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Set default parameters (matching main.py defaults)
EPOCHS=100
BATCH_SIZE=32
LR=0.0001
D_MODEL=256
N_HEADS=4
N_LAYERS=4
PATIENCE=5
MIN_IMPROVEMENT=0.00001

echo "Training parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Model dimension: $D_MODEL"
echo "  Attention heads: $N_HEADS"
echo "  Transformer layers: $N_LAYERS"
echo "  Patience: $PATIENCE"
echo "  Min improvement: $MIN_IMPROVEMENT"
echo ""

# Train baseline model
echo "Training baseline model..."
python main.py train --model baseline --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --d-model $D_MODEL --n-heads $N_HEADS --n-layers $N_LAYERS --patience $PATIENCE --min-improvement $MIN_IMPROVEMENT

# Train edit-aware models with different alpha values
ALPHA_VALUES=(0.1 0.5 1.0 2.0 3.0)

for alpha in "${ALPHA_VALUES[@]}"; do
    echo ""
    echo "Training edit-aware model with alpha=$alpha..."
    python main.py train --model edit_aware --alpha $alpha --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --d-model $D_MODEL --n-heads $N_HEADS --n-layers $N_LAYERS --patience $PATIENCE --min-improvement $MIN_IMPROVEMENT
done

echo ""
echo "Training completed! Starting evaluation..."
echo "=========================================="

# Evaluate all models
echo "Evaluating baseline model..."
python main.py evaluate --model baseline --dataset test

for alpha in "${ALPHA_VALUES[@]}"; do
    echo ""
    echo "Evaluating edit-aware model with alpha=$alpha..."
    python main.py evaluate --model edit_aware_alpha_$alpha --dataset test
done

echo ""
echo "All training and evaluation completed!"
echo "Results saved to:"
echo "  Models: models/"
echo "  Training logs: logs/*_loss_train_val.json"
echo "  Evaluation results: logs/eval_results_*.json"