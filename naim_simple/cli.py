"""
Command-line interface for NAIM Simple.
"""
import click
import pandas as pd
import numpy as np
from . import NAIM
from .utils import dataframe_to_tensors
import torch
import pickle
import os


@click.group()
def cli():
    """NAIM Simple CLI."""
    pass


@cli.command()
@click.option('--data', required=True, help='Path to training CSV file.')
@click.option('--target', required=True, help='Name of target column.')
@click.option('--cat', required=True, help='Comma-separated list of categorical column names.')
@click.option('--num', required=True, help='Comma-separated list of numerical column names.')
@click.option('--output', required=True, help='Path to save the trained model.')
@click.option('--embedding-dim', default=32, type=int, help='Dimension of feature embeddings.')
@click.option('--n-layers', default=4, type=int, help='Number of transformer encoder layers.')
@click.option('--n-heads', default=4, type=int, help='Number of attention heads.')
@click.option('--learning-rate', default=1e-3, type=float, help='Learning rate for optimizer.')
@click.option('--batch-size', default=128, type=int, help='Batch size for training.')
@click.option('--epochs', default=200, type=int, help='Maximum number of training epochs.')
@click.option('--early-stopping-patience', default=20, type=int, help='Patience for early stopping.')
@click.option('--missing-simulation/--no-missing-simulation', default=True, help='Whether to use missing simulation regularization.')
@click.option('--device', default='auto', type=click.Choice(['cpu', 'cuda', 'auto']), help='Device to use.')
def train(data, target, cat, num, output, embedding_dim, n_layers, n_heads, learning_rate, batch_size, epochs, early_stopping_patience, missing_simulation, device):
    """Train a NAIM model."""
    # Load data
    df = pd.read_csv(data)
    
    # Prepare features and target
    cat_features = [c.strip() for c in cat.split(',')]
    num_features = [n.strip() for n in num.split(',')]
    
    X = df[cat_features + num_features]
    y = df[target]
    
    # Create and train model
    model = NAIM(
        cat_features=cat_features,
        num_features=num_features,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        missing_simulation=missing_simulation,
        device=device
    )
    
    model.fit(X, y)
    
    # Save model
    model.save(output)
    click.echo(f"Model saved to {output}")


@cli.command()
@click.option('--model', required=True, help='Path to trained model file.')
@click.option('--test', required=True, help='Path to test CSV file.')
@click.option('--out', required=True, help='Path to save predictions CSV.')
@click.option('--probs/--no-probs', default=False, help='Whether to output class probabilities instead of labels.')
def predict(model, test, out, probs):
    """Make predictions with a trained NAIM model."""
    # Load model
    loaded_model = NAIM.load(model)
    
    # Load test data
    df_test = pd.read_csv(test)
    
    # Make predictions
    if probs:
        preds = loaded_model.predict_proba(df_test)
        # Save as CSV with class names as columns
        pred_df = pd.DataFrame(preds, columns=[f"class_{c}" for c in loaded_model.classes_])
    else:
        preds = loaded_model.predict(df_test)
        pred_df = pd.DataFrame({'prediction': preds})
    
    # Save predictions
    pred_df.to_csv(out, index=False)
    click.echo(f"Predictions saved to {out}")


if __name__ == '__main__':
    cli()