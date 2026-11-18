"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    so = cfg["second_opinion"]
    conf_thr = so["conf"]
    min_area_px = so["min_area_px"]
    fp_conf = so["fp_conf"]
    iou_missing = so["iou_missing"]
    iou_match = so["iou_match"]
    iou_loose = so["iou_loose"]
    num_classes = len(ds.classes)

    issues = []
    for p in ds.image_files():
        stem = p.stem
        if stem not in preds:
            continue
        raw_lines = ds.read_label_lines(stem)
        if raw_lines is None:
            continue
        img_size = ds.image_size(stem)
        if img_size is None:
            continue
        iw, ih = img_size

        label_items = []
        for line_no, text in raw_lines:
            try:
                cls_id, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if 0 <= cls_id < num_classes:
                label_items.append((line_no, cls_id, (cx, cy, w, h)))

        img_preds = preds[stem]

        for r in img_preds:
            r_cls, r_conf, rx, ry, rw, rh = r
            if r_conf < conf_thr:
                continue
            if (rw * iw) * (rh * ih) < min_area_px:
                continue
            r_box = (rx, ry, rw, rh)
            cands = [
                (l_cls, iou_xywh(r_box, l_box))
                for _, l_cls, l_box in label_items
            ]
            cands = [(l_cls, iou) for l_cls, iou in cands if iou >= iou_missing]

            if not cands:
                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_MISSING",
                        stem,
                        f"model thấy lớp {r_cls} (conf={r_conf:.2f}) nhưng không có box nhãn khớp",
                        extra={"conf": r_conf, "class": r_cls},
                    )
                )
            else:
                same_ious = [iou for l_cls, iou in cands if l_cls == r_cls]
                if same_ious:
                    best_same = max(same_ious)
                    if best_same < iou_loose:
                        issues.append(
                            Issue(
                                "second_opinion",
                                WARN,
                                "POSSIBLE_LOOSE_BOX",
                                stem,
                                f"box nhãn cùng lớp {r_cls} nhưng lỏng (IoU={best_same:.2f} < {iou_loose})",
                                extra={"conf": r_conf, "iou": best_same},
                            )
                        )
                else:
                    best_other = max(iou for _, iou in cands)
                    if best_other >= iou_match:
                        issues.append(
                            Issue(
                                "second_opinion",
                                WARN,
                                "POSSIBLE_WRONG_CLASS",
                                stem,
                                f"model dự đoán lớp {r_cls} (conf={r_conf:.2f}, IoU={best_other:.2f}) khác lớp nhãn",
                                extra={"conf": r_conf, "iou": best_other},
                            )
                        )

        fp_boxes = [
            (px, py, pw, ph)
            for _, p_conf, px, py, pw, ph in img_preds
            if p_conf >= fp_conf
        ]
        for line_no, _, l_box in label_items:
            best_iou = max((iou_xywh(l_box, p_box) for p_box in fp_boxes), default=0.0)
            if best_iou < iou_missing:
                issues.append(
                    Issue(
                        "second_opinion",
                        WARN,
                        "POSSIBLE_FALSE_POS",
                        stem,
                        f"nhãn dòng {line_no} không khớp dự đoán nào (max IoU={best_iou:.2f})",
                        line=line_no,
                        extra={"iou": best_iou},
                    )
                )

    return issues
