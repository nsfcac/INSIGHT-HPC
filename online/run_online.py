from __future__ import annotations

import argparse


# Parse CLI and drive the 60s tick loop / --once / --backfill modes.
def main() -> None:
    parser = argparse.ArgumentParser(description="INSIGHT-HPC online detector")
    parser.add_argument("--once", action="store_true", help="run a single cycle")
    parser.add_argument("--backfill", nargs=2, metavar=("START", "END"))
    parser.parse_args()
    raise NotImplementedError("online tick loop")


if __name__ == "__main__":
    main()
