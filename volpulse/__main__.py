"""``python -m volpulse [TICKER ...]`` runs the historical ETL."""

from .data.etl import main

if __name__ == "__main__":
    main()
