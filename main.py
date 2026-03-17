import argparse
import os
import json
import pickle
import torch
from pathlib import Path
from src.train import (
    create_dataloaders,
    create_model_and_optimizer,
    train_baseline,
    train_edit_aware,
    evaluate_model,
    device
)


def create_directories():
    """Create necessary directories for saving models and logs."""
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


def save_loss(loss_dict, model_name, loss_type="train"):
    """Save loss dictionary to JSON file."""
    filepath = f"logs/{model_name}_loss_{loss_type}.json"
    # Convert defaultdict to dict for JSON serialization
    loss_data = {k: v for k, v in loss_dict.items()}
    with open(filepath, 'w') as f:
        json.dump(loss_data, f, indent=2)
    print(f"Loss saved to {filepath}")


def save_model(model, model_name):
    """Save model state dict to file."""
    filepath = f"models/{model_name}.pth"
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, model_name):
    """Load model state dict from file."""
    filepath = f"models/{model_name}.pth"
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath, map_location=device))
        print(f"Model loaded from {filepath}")
        return True
    else:
        print(f"Model file {filepath} not found")
        return False


def train_command(args):
    """Execute the training command."""
    print(f"Starting training for model: {args.model}")
    print(f"Device: {device}")
    
    # Create directories
    create_directories()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader = create_dataloaders(data_name='train', batch_size=args.batch_size)
    val_dataloader = create_dataloaders(data_name='val', batch_size=args.batch_size)
    
    # Create model and optimizer
    print(f"Creating model with parameters: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    model, optimizer = create_model_and_optimizer(
        model_name=args.model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr
    )
    
    # Train based on model type
    if args.model == "baseline":
        print(f"Training baseline model for {args.epochs} epochs with min_improvement={args.min_improvement}")
        loss_dict = train_baseline(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            device_arg=device,
            epochs=args.epochs,
            patience=args.patience,
            min_improvement=args.min_improvement
        )
        model_save_name = "baseline"
    
    elif args.model == "edit_aware":
        print(f"Training edit-aware model for {args.epochs} epochs with alpha={args.alpha}, min_improvement={args.min_improvement}")
        loss_dict = train_edit_aware(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            device_arg=device,
            alpha=args.alpha,
            epochs=args.epochs,
            patience=args.patience,
            min_improvement=args.min_improvement
        )
        model_save_name = f"edit_aware_alpha_{args.alpha}"
    
    else:
        print(f"Unknown model: {args.model}")
        return
    
    # Save model and loss
    save_model(model, model_save_name)
    save_loss(loss_dict, model_save_name, "train_val")
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved to: models/{model_save_name}.pth")
    print(f"Loss history saved to: logs/{model_save_name}_loss_train_val.json")


def evaluate_command(args):
    """Execute the evaluation command."""
    print(f"Starting evaluation for model: {args.model}")
    print(f"Device: {device}")
    
    # Create directories
    create_directories()
    
    # Create dataloaders
    print("Creating dataloaders...")
    
    
    
    # Determine which dataloader to use
    if args.dataset == "validation":
        dataloader = create_dataloaders(data_name='val', batch_size=args.batch_size)
        print("Evaluating on validation dataset...")
    elif args.dataset == "test":
        dataloader = create_dataloaders(data_name='test', batch_size=args.batch_size)
        print("Evaluating on test dataset...")
    else:
        print(f"Unknown dataset: {args.dataset}")
        return
    
    # Create model
    print(f"Creating model with parameters: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    model, _ = create_model_and_optimizer(
        model_name=args.model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lr=args.lr
    )
    
    # Load model
    if not load_model(model, args.model_path):
        print(f"Error: Could not load model from {args.model_path}")
        return
    
    # Evaluate
    print("\nRunning evaluation...")
    total_cost, stats, sample_costs = evaluate_model(model, dataloader, device_arg=device)
    
    # Display results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {args.model}")
    print("="*60)
    print(f"Total edit cost: {total_cost}")
    print(f"Average cost per sample: {total_cost / len(sample_costs):.4f}")
    print(f"\nEdit statistics:")
    for stat_name, stat_value in stats.items():
        print(f"  {stat_name}: {stat_value}")
    
    if len(sample_costs) > 0:
        print(f"\nSample costs statistics:")
        print(f"  Mean: {sample_costs[:, 0].mean():.4f}")
        print(f"  Std: {sample_costs[:, 0].std():.4f}")
        print(f"  Min: {sample_costs[:, 0].min():.4f}")
        print(f"  Max: {sample_costs[:, 0].max():.4f}")
    
    print("="*60)
    
    # Save results
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "total_cost": float(total_cost),
        "average_cost_per_sample": float(total_cost / len(sample_costs)) if len(sample_costs) > 0 else 0,
        "edit_statistics": {k: int(v) for k, v in stats.items()},
        "sample_costs": {
            "mean": float(sample_costs[:, 0].mean()),
            "std": float(sample_costs[:, 0].std()),
            "min": float(sample_costs[:, 0].min()),
            "max": float(sample_costs[:, 0].max())
        }
    }
    
    results_file = f"logs/eval_results_{args.model}_{args.dataset}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Music Edit Learning Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --model baseline
  python main.py train --model edit_aware --alpha 0.5
  python main.py train --model baseline --epochs 100 --batch-size 16
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "edit_aware"],
        required=True,
        help="Model to train (baseline or edit_aware)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)"
    )
    train_parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension (default: 256)"
    )
    train_parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)"
    )
    train_parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 4)"
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)"
    )
    train_parser.add_argument(
        "--min-improvement",
        type=float,
        default=1e-3,
        help="Minimum improvement threshold (default: 1e-3)"
    )
    train_parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Edit weight factor for edit_aware model (default: 0.3)"
    )
    train_parser.set_defaults(func=train_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to evaluate (e.g., baseline, edit_aware)"
    )
    eval_parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model file (e.g., baseline, edit_aware_alpha_0.3). If not specified, uses --model value"
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        choices=["validation", "test"],
        default="test",
        help="Dataset to evaluate on (default: test)"
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    eval_parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension (default: 256)"
    )
    eval_parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)"
    )
    eval_parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 4)"
    )
    eval_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    eval_parser.set_defaults(func=evaluate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set model_path if not specified
    if hasattr(args, "model_path") and args.model_path is None:
        args.model_path = args.model
    
    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
