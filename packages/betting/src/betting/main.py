import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

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
