#!/usr/bin/env python3
"""Format and display VF comparison report in a beautiful terminal or markdown table."""

import argparse
import json
from pathlib import Path
import sys


def print_table(data, baseline_name="Tập 1: Gốc VF6", refined_name="Tập 2: Reproject", as_markdown=False):
    b = data.get("baseline", {})
    r = data.get("refined", {})
    if not b or not r:
        print("[LỖI] File JSON không đúng cấu trúc (thiếu baseline hoặc refined).", file=sys.stderr)
        return

    d_map = (r.get("custom_mAP", 0.0) - b.get("custom_mAP", 0.0)) * 100

    classes = [c for c in ["car", "motorcycle", "pedestrian"] if c in b.get("per_class", {})]

    if as_markdown:
        print(f"\n### Bảng So Sánh: {baseline_name} vs {refined_name}\n")
        print(f"**custom_mAP**: {baseline_name} = **{b.get('custom_mAP', 0.0)*100:0.2f}%** | "
              f"{refined_name} = **{r.get('custom_mAP', 0.0)*100:0.2f}%** | "
              f"Chênh lệch = **{d_map:+0.2f}%**\n")
        print("| Lớp vật thể | Chỉ số | " + f"{baseline_name} | {refined_name} | Mức cải thiện |")
        print("| :--- | :--- | :---: | :---: | :---: |")

        for cls in classes:
            bp = b["per_class"][cls]
            rp = r["per_class"][cls]
            gt_cnt = bp.get("gt_boxes", 0)
            cls_label = f"**{cls.upper()}** ({gt_cnt})"

            d_ap = (rp["AP"] - bp["AP"]) * 100
            mult = f" (x{rp['AP']/bp['AP']:0.1f})" if bp["AP"] > 0 and rp["AP"]/bp["AP"] >= 1.5 else ""
            print(f"| {cls_label} | **AP trung bình** | **{bp['AP']*100:0.2f}%** | **{rp['AP']*100:0.2f}%** | **{d_ap:+0.2f}%**{mult} |")

            for dist in ["0.5", "1.0", "2.0", "4.0"]:
                b_d = bp.get("AP_by_distance_m", {}).get(dist, 0.0) * 100
                r_d = rp.get("AP_by_distance_m", {}).get(dist, 0.0) * 100
                d_d = r_d - b_d
                print(f"| | └─ AP @ {dist}m | {b_d:0.2f}% | {r_d:0.2f}% | {d_d:+0.2f}% |")

            err_configs = [
                ("trans_err", "Sai số vị trí (mATE)", "m"),
                ("scale_err", "Sai số kích thước (mASE)", "1-IoU"),
                ("orient_err", "Sai số góc yaw (mAOE)", "rad"),
            ]
            for key, name, unit in err_configs:
                bv = bp.get(key, 0.0)
                rv = rp.get(key, 0.0)
                diff = rv - bv
                pct = (diff / bv * 100) if bv > 0 else 0.0
                status = "Tốt hơn" if diff < 0 else ("Kém hơn" if diff > 0 else "Bằng")

                if key == "orient_err":
                    b_str = f"{bv:.4f} ({bv * 180 / 3.14159265:.1f}°)"
                    r_str = f"{rv:.4f} ({rv * 180 / 3.14159265:.1f}°)"
                elif key == "trans_err":
                    b_str = f"{bv:.4f} m"
                    r_str = f"{rv:.4f} m"
                else:
                    b_str = f"{bv:.4f}"
                    r_str = f"{rv:.4f}"

                print(f"| | {name} | {b_str} | {r_str} | {pct:+0.1f}% ({status}) |")
        print()
        return

    # Unicode Box Drawing Format
    W = 100
    print("\n┌" + "─"*W + "┐")
    title = f" BẢNG SO SÁNH HIỆU NĂNG: {baseline_name} vs {refined_name} "
    print("│" + title.center(W) + "│")
    print("├" + "─"*W + "┤")

    map_str = (f" custom_mAP: {baseline_name} = {b.get('custom_mAP', 0.0)*100:6.2f}%  │  "
               f"{refined_name} = {r.get('custom_mAP', 0.0)*100:6.2f}%  │  Chênh lệch = {d_map:+6.2f}% ")
    print("│" + map_str.center(W) + "│")
    print("├" + "─"*16 + "┬" + "─"*26 + "┬" + "─"*18 + "┬" + "─"*20 + "┬" + "─"*18 + "┤")
    print(f"│ {'Lớp vật thể':<14} │ {'Chỉ số':<24} │ {baseline_name:<16} │ {refined_name:<18} │ {'Mức cải thiện':<16} │")
    print("├" + "─"*16 + "┼" + "─"*26 + "┼" + "─"*18 + "┼" + "─"*20 + "┼" + "─"*18 + "┤")

    for c_idx, cls in enumerate(classes):
        bp = b["per_class"][cls]
        rp = r["per_class"][cls]
        gt_cnt = bp.get("gt_boxes", 0)
        cls_label = f"{cls.upper()} ({gt_cnt})"

        # AP metrics
        d_ap = (rp["AP"] - bp["AP"]) * 100
        mult = f" (x{rp['AP']/bp['AP']:0.1f})" if bp["AP"] > 0 and rp["AP"]/bp["AP"] >= 1.5 else ""
        print(f"│ {cls_label:<14} │ {'AP trung bình':<24} │ {bp['AP']*100:6.2f}%          │ {rp['AP']*100:6.2f}%            │ {d_ap:+6.2f}%{mult:<9} │")

        # Distance APs
        for dist in ["0.5", "1.0", "2.0", "4.0"]:
            b_d = bp.get("AP_by_distance_m", {}).get(dist, 0.0) * 100
            r_d = rp.get("AP_by_distance_m", {}).get(dist, 0.0) * 100
            d_d = r_d - b_d
            print(f"│ {'':<14} │ {f'  ├─ AP @ {dist}m':<24} │ {b_d:6.2f}%          │ {r_d:6.2f}%            │ {d_d:+6.2f}%           │")

        # Error metrics (ATE, ASE, AOE)
        err_configs = [
            ("trans_err", "Sai số vị trí (mATE)", "m"),
            ("scale_err", "Sai số kích thước (mASE)", "1-IoU"),
            ("orient_err", "Sai số góc yaw (mAOE)", "rad"),
        ]
        for key, name, unit in err_configs:
            bv = bp.get(key, 0.0)
            rv = rp.get(key, 0.0)
            diff = rv - bv
            pct = (diff / bv * 100) if bv > 0 else 0.0
            status = "Tốt hơn" if diff < 0 else ("Kém hơn" if diff > 0 else "Bằng")

            if key == "orient_err":
                b_deg = bv * 180 / 3.14159265
                r_deg = rv * 180 / 3.14159265
                b_str = f"{bv:.4f} ({b_deg:4.1f}°)"
                r_str = f"{rv:.4f} ({r_deg:4.1f}°)"
            elif key == "trans_err":
                b_str = f"{bv:.4f} m"
                r_str = f"{rv:.4f} m"
            else:
                b_str = f"{bv:.4f}"
                r_str = f"{rv:.4f}"

            diff_str = f"{pct:+5.1f}% ({status})"
            print(f"│ {'':<14} │ {name:<24} │ {b_str:<16} │ {r_str:<18} │ {diff_str:<16} │")

        if c_idx < len(classes) - 1:
            print("├" + "─"*16 + "┼" + "─"*26 + "┼" + "─"*18 + "┼" + "─"*20 + "┼" + "─"*18 + "┤")

    print("└" + "─"*16 + "┴" + "─"*26 + "┴" + "─"*18 + "┴" + "─"*20 + "┴" + "─"*18 + "┘\n")


def main():
    parser = argparse.ArgumentParser(description="In bảng so sánh metric VF từ file JSON.")
    parser.add_argument("json_path", nargs="?", default="outputs/vf6_nav/compare_reproject_sweeps2.json",
                        help="Đường dẫn file JSON do evaluate_vf_comparison.py xuất ra")
    parser.add_argument("--baseline-name", default="Tập 1: Gốc VF6", help="Tên hiển thị tập baseline")
    parser.add_argument("--refined-name", default="Tập 2: Reproject", help="Tên hiển thị tập refined")
    parser.add_argument("--markdown", action="store_true", help="Xuất định dạng bảng GitHub Markdown")
    args = parser.parse_args()

    json_file = Path(args.json_path)
    if not json_file.exists():
        print(f"[LỖI] Không tìm thấy file: {json_file}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(json_file.read_text())
    print_table(data, baseline_name=args.baseline_name, refined_name=args.refined_name, as_markdown=args.markdown)


if __name__ == "__main__":
    main()
