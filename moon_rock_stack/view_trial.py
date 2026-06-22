from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a generated MuJoCo rock stacking trial.")
    parser.add_argument("xml", type=Path, help="Path to generated MJCF XML.")
    args = parser.parse_args()

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
