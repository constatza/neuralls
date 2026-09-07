"""Public root CLI for neuralls."""

from __future__ import annotations

import typer

from neuralls.cli.compare import compare_case
from neuralls.cli.config import app as config_app
from neuralls.cli.eval import eval_case_batch
from neuralls.cli.generate import generate_case
from neuralls.cli.generate_single import generate_single
from neuralls.cli.run import run_case_pipeline_command
from neuralls.cli.train import train_case_batch

app = typer.Typer(
    help="Batch-oriented neural preconditioner workflows.",
    no_args_is_help=True,
)

app.add_typer(config_app, name="config")
app.command("generate")(generate_case)
app.command("generate-single")(generate_single)
app.command("train")(train_case_batch)
app.command("eval")(eval_case_batch)
app.command("run")(run_case_pipeline_command)
app.command("compare")(compare_case)
