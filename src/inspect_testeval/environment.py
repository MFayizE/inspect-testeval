"""Docker sandbox used to run model-generated tests.

The compose file is resolved relative to this module so the right image is
used whatever the caller's working directory is.
"""

from pathlib import Path

SANDBOX_DIR = Path(__file__).parent / "sandbox"
SANDBOX_COMPOSE_FILE = SANDBOX_DIR / "compose.yaml"
SANDBOX = ("docker", str(SANDBOX_COMPOSE_FILE))
