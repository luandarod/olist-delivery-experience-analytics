"""CLI wrapper for the Olist analytics pipeline."""

from __future__ import annotations

from _bootstrap import bootstrap_src_path

bootstrap_src_path()

from olist_delivery_experience_analytics.cli import main


if __name__ == "__main__":
    main()
