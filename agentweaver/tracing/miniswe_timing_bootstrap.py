from __future__ import annotations

import sys

import agentweaver.tracing.miniswe_timing_patch  # noqa: F401
from minisweagent.run.utilities.mini_extra import main as mini_extra_main


def main() -> None:
    sys.argv = ["mini-extra", *sys.argv[1:]]
    raise SystemExit(mini_extra_main())


if __name__ == "__main__":
    main()
