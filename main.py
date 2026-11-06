"""Bytebuddies NHL Predictor - Main Entry Point."""

from src.bronze_ingestion import main as run_bronze_cli


def main():
    """Main CLI entrypoint."""
    run_bronze_cli()


if __name__ == "__main__":
    main()
