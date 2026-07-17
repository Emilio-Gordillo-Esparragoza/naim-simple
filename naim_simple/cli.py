"""Command-line interface for NAIM Simple."""

from pathlib import Path
from typing import Iterable, List, Optional

import click
import pandas as pd

from .core import NAIM


def _columns(value: str) -> List[str]:
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise click.BadParameter("at least one column is required")
    return columns


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise click.ClickException(f"{source} is missing columns: {sorted(missing)}")


@click.group()
def cli() -> None:
    """Train and run NAIM Simple models."""


@cli.command()
@click.option("--data", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--target", required=True, help="Name of the target column.")
@click.option("--cat", required=True, help="Comma-separated categorical columns.")
@click.option("--num", required=True, help="Comma-separated numerical columns.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for the saved model bundle.",
)
@click.option(
    "--validation-data",
    type=click.Path(path_type=Path, exists=True),
    help="Optional validation CSV used for early stopping.",
)
@click.option(
    "--validation-target",
    help="Validation target column; defaults to --target.",
)
@click.option("--embedding-dim", default=32, type=click.IntRange(min=1), show_default=True)
@click.option("--n-layers", default=4, type=click.IntRange(min=1), show_default=True)
@click.option("--n-heads", default=4, type=click.IntRange(min=1), show_default=True)
@click.option("--learning-rate", default=1e-3, type=click.FloatRange(min=0, min_open=True))
@click.option("--batch-size", default=128, type=click.IntRange(min=1), show_default=True)
@click.option("--epochs", default=200, type=click.IntRange(min=1), show_default=True)
@click.option(
    "--early-stopping-patience",
    default=20,
    type=click.IntRange(min=1),
    show_default=True,
)
@click.option("--missing-simulation/--no-missing-simulation", default=True)
@click.option("--device", default="auto", type=click.Choice(["cpu", "cuda", "auto"]))
@click.option("--random-state", type=int)
def train(
    data: Path,
    target: str,
    cat: str,
    num: str,
    output: Path,
    validation_data: Optional[Path],
    validation_target: Optional[str],
    embedding_dim: int,
    n_layers: int,
    n_heads: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    early_stopping_patience: int,
    missing_simulation: bool,
    device: str,
    random_state: Optional[int],
) -> None:
    """Train a NAIM model."""
    frame = pd.read_csv(data)
    cat_features = _columns(cat)
    num_features = _columns(num)
    feature_names = cat_features + num_features
    _require_columns(frame, feature_names + [target], data)

    X_val = None
    y_val = None
    if validation_data is not None:
        validation_frame = pd.read_csv(validation_data)
        validation_target = validation_target or target
        _require_columns(validation_frame, feature_names + [validation_target], validation_data)
        X_val = validation_frame[feature_names]
        y_val = validation_frame[validation_target]
    elif validation_target is not None:
        raise click.ClickException("--validation-target requires --validation-data")

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
        device=device,
        random_state=random_state,
    )
    model.fit(frame[feature_names], frame[target], X_val=X_val, y_val=y_val)
    model.save(output)
    click.echo(f"Model saved to {output}")


@cli.command()
@click.option(
    "--model",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option("--test", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--probs/--no-probs", default=False)
@click.option("--device", default="auto", type=click.Choice(["cpu", "cuda", "auto"]))
def predict(model: Path, test: Path, out: Path, probs: bool, device: str) -> None:
    """Make predictions with a saved NAIM model."""
    loaded_model = NAIM.load(model, device=device)
    frame = pd.read_csv(test)
    feature_names = loaded_model.feature_names_ or []
    _require_columns(frame, feature_names, test)
    features = frame[feature_names]

    if probs:
        values = loaded_model.predict_proba(features)
        result = pd.DataFrame(values, columns=[f"class_{value}" for value in loaded_model.classes_])
    else:
        result = pd.DataFrame({"prediction": loaded_model.predict(features)})
    result.to_csv(out, index=False)
    click.echo(f"Predictions saved to {out}")


if __name__ == "__main__":
    cli()
