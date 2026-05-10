"""
Apply the three-stage hierarchical bucketing pipeline to a CSV + embedding
matrix and write the grouped CSV plus four summary tables.

This is a thin wrapper around :func:`m2p.grouping.group_dataset`.

Example
-------
.. code-block:: bash

    python scripts/run_grouping.py \
        --csv data/final_dataset3.csv \
        --emb data/features/all_pooled_last.npy \
        --out_dir grouping_output
"""

from __future__ import annotations

from m2p.grouping import main as grouping_main


if __name__ == "__main__":
    grouping_main()
