import json
from pathlib import Path


def compare_metrics(
    orig_path="outputs/mini-eval/metrics_summary.json",
    ref_path="outputs/detzero_refined/eval_detection/metrics_summary.json"
):
    orig_file = Path(orig_path)
    ref_file = Path(ref_path)

    if not orig_file.exists():
        print(f"[LỖI] Không tìm thấy file gốc: {orig_file}")
        return
    if not ref_file.exists():
        print(f"[LỖI] Không tìm thấy file sau refine: {ref_file}")
        return

    orig = json.loads(orig_file.read_text())
    ref = json.loads(ref_file.read_text())

    print("\n" + "=" * 78)
    print("      BẢNG SO SÁNH: BEVFUSION GỐC vs BEVFUSION + DETZERO REFINED")
    print("=" * 78)

    o_map, r_map = orig["mean_ap"] * 100, ref["mean_ap"] * 100
    o_nds, r_nds = orig["nd_score"] * 100, ref["nd_score"] * 100
    d_map = r_map - o_map
    d_nds = r_nds - o_nds

    print(f"\n1. CHỈ SỐ TỔNG QUAN:")
    print(f"  * mAP:  BEVFusion Gốc = {o_map:6.2f}% | Sau Refine = {r_map:6.2f}% | Chênh lệch = {d_map:+5.2f}%")
    print(f"  * NDS:  BEVFusion Gốc = {o_nds:6.2f}% | Sau Refine = {r_nds:6.2f}% | Chênh lệch = {d_nds:+5.2f}%")

    print(f"\n2. TỔNG HỢP SAI SỐ TRUNG BÌNH VỚI GROUND TRUTH (mATE, mASE, mAOE):")
    tp_orig = orig["tp_errors"]
    tp_ref = ref["tp_errors"]

    print(f"  {'Chỉ số sai số':<26} | {'BEVFusion Gốc':<15} | {'Sau DetZero':<15} | {'Cải thiện'}")
    print("  " + "-" * 74)
    labels_map = {
        "trans_err": ("Sai số vị trí tâm (mATE)", "m"),
        "orient_err": ("Sai số góc quay (mAOE)", "rad"),
        "scale_err": ("Sai số kích thước (mASE)", "1-iou"),
        "vel_err": ("Sai số vận tốc (mAVE)", "m/s"),
    }
    for k, (name, unit) in labels_map.items():
        vo = tp_orig.get(k, 0.0)
        vr = tp_ref.get(k, 0.0)
        diff_pct = (vr - vo) / vo * 100 if vo > 0 else 0.0
        status = "Tốt hơn" if diff_pct < 0 else ("Kém hơn" if diff_pct > 0 else "Bằng")
        print(f"  {name:<26} | {vo:7.4f} {unit:<7} | {vr:7.4f} {unit:<7} | {diff_pct:+6.2f}% ({status})")

    print(f"\n3. CHI TIẾT THEO TỪNG LỚP VẬT THỂ (AP, Vị trí mATE, Góc quay mAOE):")
    print(f"  {'Lớp':<12} | {'Chỉ số':<14} | {'BEVFusion':<12} | {'Sau Refine':<12} | {'Mức thay đổi'}")
    print("  " + "-" * 74)

    classes = ["car", "truck", "bus", "pedestrian", "motorcycle", "bicycle"]
    for c in classes:
        o_ap = orig["mean_dist_aps"].get(c, 0.0) * 100
        r_ap = ref["mean_dist_aps"].get(c, 0.0) * 100
        d_ap = r_ap - o_ap

        o_ate = orig["label_tp_errors"][c]["trans_err"]
        r_ate = ref["label_tp_errors"][c]["trans_err"]
        d_ate = (r_ate - o_ate) / o_ate * 100

        o_aoe = orig["label_tp_errors"][c]["orient_err"]
        r_aoe = ref["label_tp_errors"][c]["orient_err"]
        d_aoe = (r_aoe - o_aoe) / o_aoe * 100
        o_ase = orig["label_tp_errors"][c]["scale_err"]
        r_ase = ref["label_tp_errors"][c]["scale_err"]
        d_ase = (r_ase - o_ase) / o_ase * 100

        print(f"  {c.upper():<12} | AP (Độ chuẩn)  | {o_ap:6.2f}%      | {r_ap:6.2f}%      | {d_ap:+5.2f}%")
        print(f"  {'':<12} | mATE (Vị trí)  | {o_ate:6.4f} m    | {r_ate:6.4f} m    | {d_ate:+5.2f}% {'(Cải thiện)' if d_ate < 0 else ''}")
        print(f"  {'':<12} | mAOE (Góc yaw) | {o_aoe:6.4f} rad  | {r_aoe:6.4f} rad  | {d_aoe:+5.2f}% {'(Cải thiện)' if d_aoe < 0 else ''}")
        print(f"  {'':<12} | mASE (K.Thước) | {o_ase:6.4f} iou  | {r_ase:6.4f} iou  | {d_ase:+5.2f}% {'(Cải thiện)' if d_ase < 0 else ''}")
        print("  " + "-" * 74)

    print("=" * 78 + "\n")


if __name__ == "__main__":
    compare_metrics()
