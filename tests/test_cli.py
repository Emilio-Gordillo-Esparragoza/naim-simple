import pandas as pd
from click.testing import CliRunner

from naim_simple.cli import cli, predict, train


def test_cli_help_and_direct_entry_points():
    runner = CliRunner()

    assert runner.invoke(cli, ["--help"]).exit_code == 0
    assert runner.invoke(train, ["--help"]).exit_code == 0
    assert runner.invoke(predict, ["--help"]).exit_code == 0


def test_train_and_predict_commands(sample_frame, tmp_path):
    runner = CliRunner()
    training_csv = tmp_path / "training.csv"
    prediction_csv = tmp_path / "prediction.csv"
    model_path = tmp_path / "model.naim"
    output_csv = tmp_path / "output.csv"
    sample_frame.to_csv(training_csv, index=False)
    sample_frame.assign(extra="ignored").to_csv(prediction_csv, index=False)

    trained = runner.invoke(
        train,
        [
            "--data",
            str(training_csv),
            "--target",
            "target",
            "--cat",
            "category",
            "--num",
            "value",
            "--output",
            str(model_path),
            "--embedding-dim",
            "4",
            "--n-heads",
            "1",
            "--n-layers",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "3",
            "--device",
            "cpu",
            "--random-state",
            "7",
        ],
    )
    assert trained.exit_code == 0, trained.output

    predicted = runner.invoke(
        predict,
        [
            "--model",
            str(model_path),
            "--test",
            str(prediction_csv),
            "--out",
            str(output_csv),
            "--device",
            "cpu",
        ],
    )
    assert predicted.exit_code == 0, predicted.output
    assert len(pd.read_csv(output_csv)) == len(sample_frame)
