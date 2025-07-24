import asyncio
import csv
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from betting.gym.evaluate import evaluate
from betting.gym.train import train_blackjack_agent

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAVE_DIR = SCRIPT_DIR / "experiments"

console = Console()


def setup_signal_handlers():
    def signal_handler(sig, frame):
        console.print("\n[yellow]Interrupted by user. Cleaning up...[/yellow]")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                for task in asyncio.all_tasks(loop):
                    task.cancel()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


@click.group()
def betting():
    pass


@betting.group()
def gym():
    pass


@gym.command(name="train")
def train_betting_agent():
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    save_dir_path = DEFAULT_SAVE_DIR / timestamp
    models_path = save_dir_path / "models"
    logs_path = save_dir_path / "logs"
    tensorboard_logs_path = save_dir_path / "tensorboard_logs"

    os.makedirs(save_dir_path, exist_ok=True)
    # Create directories if they don't exist
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(tensorboard_logs_path, exist_ok=True)

    # Train the agent
    train_blackjack_agent(
        models_path=models_path,
        logs_path=logs_path,
        tensoboard_logs_path=tensorboard_logs_path,
    )


@gym.command(name="eval")
@click.option(
    "--model-path", type=click.Path(exists=True), help="Path to the model file"
)
def evaluation_model(model_path):
    if model_path is None:
        # Find the latest experiment directory
        if not DEFAULT_SAVE_DIR.exists():
            console.print("[red]No experiments directory found[/red]")
            sys.exit(1)

        experiment_dirs = [d for d in DEFAULT_SAVE_DIR.iterdir() if d.is_dir()]
        if not experiment_dirs:
            console.print("[red]No experiment directories found[/red]")
            sys.exit(1)

        latest_dir = max(experiment_dirs, key=lambda d: d.stat().st_mtime)
        models_dir = latest_dir / "models"

        if not models_dir.exists():
            console.print(f"[red]No models directory found in {latest_dir}[/red]")
            sys.exit(1)

        model_files = list(models_dir.glob("*.zip"))
        if not model_files:
            console.print(f"[red]No model files found in {models_dir}[/red]")
            sys.exit(1)

        model_path = max(model_files, key=lambda f: f.stat().st_mtime)
        console.print(f"[green]Using latest model: {model_path}[/green]")
    mean_reward, mean_std = evaluate(model_path)
    console.print(f"Model mean reward: {mean_reward} ± {mean_std}")
    # Save results to CSV
    results_path = latest_dir / "results" / "results.csv"
    os.makedirs(results_path.parent, exist_ok=True)

    with open(results_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model_path", "mean_reward", "std_deviation"])
        writer.writerow([str(model_path), mean_reward, mean_std])

    console.print(f"[green]Results saved to: {results_path}[/green]")


def main():
    setup_signal_handlers()

    try:
        betting()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger = logging.Logger(name="train")
        logger.exception(
            msg=f"[red]Error: {e}[/red]",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
