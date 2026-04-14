import json
import sys

bbox_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/LabelAny3D/experimental_results/single/val/input/3dbbox.json"

with open(bbox_path) as f:
    bboxes = json.load(f)

print(f"{'obj_id':<8} {'dx(w)':>8} {'dy(h)':>8} {'dz(d)':>8} {'volume(m³)':>12}")
print("-"*52)

total_volume = 0.0
for obj in bboxes:
    d = obj["dimensions"]
    oid = obj["obj_id"]
    vol = d[0] * d[1] * d[2]
    total_volume += vol
    print(f"{oid:<8} {d[0]:>8.3f} {d[1]:>8.3f} {d[2]:>8.3f} {vol:>12.4f}")

print("-"*52)
print(f"{'합계':<28} {total_volume:>12.4f} m³")
print(f"총 객체 수: {len(bboxes)}개")