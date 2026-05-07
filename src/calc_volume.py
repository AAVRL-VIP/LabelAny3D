import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate object volumes from 3dbbox.json and recommend truck tonnage."
    )
    parser.add_argument(
        "bbox_path",
        nargs="?",
        default="/workspace/LabelAny3D/experimental_results/single/val/input/3dbbox.json",
        help="Path to 3dbbox.json",
    )
    parser.add_argument(
        "--one-ton-capacity-m3",
        type=float,
        default=6.0,
        help="Reference cargo volume for a 1-ton truck (m^3). Default: 6.0",
    )
    parser.add_argument(
        "--fill-rate",
        type=float,
        default=0.8,
        help="Conservative packing factor in [0, 1]. Default: 0.8",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bbox_path = Path(args.bbox_path)
    if not bbox_path.exists():
        raise FileNotFoundError(f"bbox file not found: {bbox_path}")
    if args.one_ton_capacity_m3 <= 0:
        raise ValueError("--one-ton-capacity-m3 must be > 0")
    if not (0 < args.fill_rate <= 1):
        raise ValueError("--fill-rate must be in (0, 1]")

    with bbox_path.open(encoding="utf-8") as f:
        bboxes = json.load(f)

    print(f"{'obj_id':<20} {'dx(m)':>8} {'dy(m)':>8} {'dz(m)':>8} {'volume(m^3)':>12}")
    print("-" * 64)

    total_volume = 0.0
    for obj in bboxes:
        dims = obj.get("dimensions", [0.0, 0.0, 0.0])
        if len(dims) != 3:
            continue
        dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
        oid = str(obj.get("obj_id", "unknown"))
        vol = dx * dy * dz
        total_volume += vol
        print(f"{oid:<20} {dx:>8.3f} {dy:>8.3f} {dz:>8.3f} {vol:>12.4f}")

    print("-" * 64)
    print(f"{'TOTAL':<44} {total_volume:>12.4f} m^3")
    print(f"{'OBJECT COUNT':<44} {len(bboxes):>12d}")
    print()

    print(
        f"Truck check (1-ton ref={args.one_ton_capacity_m3:.2f} m^3, fill_rate={args.fill_rate:.2f})"
    )
    print(f"{'truck':<8} {'nominal(m^3)':>14} {'effective(m^3)':>15} {'fit':>8}")
    print("-" * 52)

    recommended_ton = None
    for ton in range(1, 11):
        nominal = args.one_ton_capacity_m3 * ton
        effective = nominal * args.fill_rate
        fit = total_volume <= effective
        if fit and recommended_ton is None:
            recommended_ton = ton
        print(f"{ton:>2} ton{'':<3} {nominal:>14.2f} {effective:>15.2f} {str(fit):>8}")

    print("-" * 52)
    if recommended_ton is not None:
        print(f"RECOMMENDED TRUCK: {recommended_ton} ton")
    else:
        effective_10t = args.one_ton_capacity_m3 * 10 * args.fill_rate
        needed_trucks = math.ceil(total_volume / effective_10t) if effective_10t > 0 else math.inf
        print("RECOMMENDED TRUCK: >10 ton class or multiple trucks")
        print(f"MIN 10-ton TRUCKS NEEDED: {needed_trucks}")


if __name__ == "__main__":
    main()
