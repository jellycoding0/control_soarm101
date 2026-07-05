import runpy
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "1_4_0_공통_오프셋_저장.py"
OUTPUT = Path(__file__).resolve().parent.parent / "config" / "joint_offsets_resting.json"


if __name__ == "__main__":
    sys.argv = [
        str(SCRIPT),
        "--pose-name",
        "resting_home",
        "--output",
        str(OUTPUT),
        *sys.argv[1:],
    ]
    runpy.run_path(str(SCRIPT), run_name="__main__")
