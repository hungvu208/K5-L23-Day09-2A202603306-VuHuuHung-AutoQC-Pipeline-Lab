# Báo cáo Day 09 — Vũ Hữu Hưng, 2A202603306

## 1. Kết luận trước
Pipeline kết luận **HOLD** (tỉ lệ ảnh còn cảnh báo sau duyệt là **18.72% = 38/203**, vượt ngưỡng `warn_ratio_hold = 10.00%`).
Trong **224** ảnh đầu vào (`data/raw`), bước nhận dữ liệu loại **21** ảnh lỗi nặng/trùng/rò rỉ, còn **203** ảnh đi qua vòng sửa tự động và duyệt mô phỏng (80 ảnh); bản xuất `submission/export/clean_v1` mới là **bản nháp thử nghiệm**, chưa đủ điều kiện phát hành chính thức.

## 2. Lỗi và phân việc
Số liệu lấy từ `submission/reports/02_error_list_v1.txt` (trước sửa; 224 ảnh thô / 203 ảnh qua bước 1):

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| `BAD_CLASS_NAME` | 2 dòng | Máy (`rename_classes`) | Tên lớp trong `data.yaml` sai hoa/thường (`Person`, `Bus`), chuẩn hoá theo `cfg["classes"]`. |
| `CORRUPT_IMAGE` | 2 ảnh | Máy (`remove_image`) | Ảnh hỏng không mở được, không thể dùng train. |
| `BAD_SIZE` | 3 ảnh | Máy (`remove_image`) | Cạnh ngắn < 250px, vi phạm quy chuẩn đầu vào. |
| `DUP_IMAGE_EXACT` | 8 ảnh | Máy (`remove_image`) | Trùng từng byte với ảnh khác trong lô, loại bản dư để tránh lệch phân phối. |
| `DUP_IMAGE_NEAR` | 3 ảnh | Máy (`remove_image`) | Trùng gần đúng (`phash_dist <= 8`), loại bản dư. |
| `LEAK_TEST` | 5 ảnh | Máy (`remove_image`) | Rò rỉ từ `data/test/images`, bắt buộc loại để đánh giá trung thực. |
| `BAD_FORMAT` | 3 dòng | Máy (`drop_line`) | Thiếu cột hoặc `nan/inf`, không phục hồi được toạ độ. |
| `BAD_CLASS_ID` | 6 dòng | Máy (`drop_line`) | ID lớp ngoài `0..3` (`-1`, `4`, `7`, `93`), xoá dòng sai. |
| `BOX_ZERO` | 4 dòng | Máy (`drop_line`) | Chiều rộng hoặc cao bằng `0`, box vô nghĩa. |
| `DUP_BOX` | 7 dòng | Máy (`drop_line`) | Box trùng lặp cùng lớp với IoU > 0.9, bỏ dòng sau. |
| `BOX_OUT_OF_IMAGE` | 8 dòng | Máy (`clip_box`) | Mép box tràn nhẹ ngoài `[0, 1]`, cắt về biên hợp lệ. |
| `MISSING_LABEL_FILE` | 6 ảnh | Người (`needs_label`) | Máy không được tự bịa nhãn; chuyển hàng đợi gán nhãn. |
| `CLASS_RARE` | 4 dòng | Thông tin (`stats`) | Nhãn lớp ngoài (`-1`, `4`, `7`) trước autofix, biến mất sau `drop_line`. |
| `BOX_TINY` | 6 dòng | Người (hàng đợi) | Box cạnh ngắn < 12px, cần người xem là vật thật hay nhiễu. |
| `BOX_HUGE` | 3 dòng | Người (hàng đợi) | Box chiếm > 95% diện tích ảnh, cần người kiểm tra. |
| `EMPTY_LABEL` | 12 ảnh | Người (hàng đợi) | File nhãn rỗng: cần phân biệt ảnh nền thật hay sót nhãn. |
| `POSSIBLE_MISSING` | 53 cảnh báo | Người (hàng đợi) | Model lớn thấy vật (`conf >= 0.6`) nhưng nhãn không có box. |
| `POSSIBLE_FALSE_POS` | 29 cảnh báo | Người (hàng đợi) | Nhãn không khớp dự đoán nào (`conf >= 0.1`), có thể nhãn dư. |
| `POSSIBLE_LOOSE_BOX` | 28 cảnh báo | Người (hàng đợi) | Đúng lớp nhưng box lỏng (`IoU < 0.5`). |
| `POSSIBLE_WRONG_CLASS` | 18 cảnh báo | Người (hàng đợi) | Trùng vị trí (`IoU >= 0.5`) nhưng khác lớp với model lớn. |

**Giải thích hàng đợi (`05_review_queue.csv`) và ngân sách 80 ảnh:**
Hàng đợi gồm **125** ảnh có cảnh báo/cần gán nhãn, xếp giảm dần theo tổng trọng số trong `cfg["review"]["weights"]` cộng `conf` của model: các ảnh có `NEEDS_LABEL` (trọng số 10, như `img_0031`..`img_0219`) và ảnh dính nhiều lỗi nặng (`img_0020` có `FIXED_DROP_LINE + POSSIBLE_MISSING + POSSIBLE_WRONG_CLASS`, priority `11.524`) được đẩy lên đầu. Với ngân sách **80/125** ảnh, người duyệt xử lý hết toàn bộ 6 ảnh `MISSING_LABEL_FILE` và các ảnh lỗi ngữ nghĩa nghiêm trọng nhất; tuy nhiên vẫn còn **45** ảnh cuối hàng đợi chưa được duyệt (trong đó **38** ảnh còn cảnh báo ở vòng 3), khiến tỉ lệ cảnh báo tồn đọng đạt **18.72%**.

## 3. Calibration (2 ngưỡng)
Số liệu sweep trên `data/calib/` (39 ảnh, `submission/calibration/calibration_report.md`):

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---:|---:|---|---|
| `duplicates.phash_dist` | 2 | **8** | `DUP_IMAGE_NEAR`: `1.00 / 0.50` → `1.00 / 0.67` (`DUP_IMAGE_EXACT` giữ `1.00 / 1.00`) | `phash_dist=8` tăng recall gần trùng lên 0.67 mà vẫn giữ precision 1.00, tránh rủi ro xoá nhầm ảnh khác nhau như khi đẩy lên 20 (recall exact tụt còn 0.33). |
| `second_opinion.conf` | 0.25 | **0.6** | `WRONG_CLASS`: `0.56/0.83` → `0.62/0.83`; `MISSING`: `0.50/1.00` → `0.57/1.00`; `LOOSE_BOX`: `0.29/0.40` → `0.20/0.20` | Tại `0.6`, recall của `MISSING` (1.00) và `WRONG_CLASS` (0.83) vẫn giữ nguyên như `0.25`, trong khi precision tăng rõ rệt, giảm báo động giả để tiết kiệm ngân sách duyệt 80 ảnh (nếu tăng lên `0.7`–`0.8` sẽ bắt đầu sót `MISSING`). |

**Một chỗ calibration có thể đánh lừa:**
Tập `data/calib/` chỉ có **39 ảnh** với số lỗi chèn rất ít (ví dụ `LOOSE_BOX` chỉ có 5 ca, `DUP_IMAGE_NEAR` 6 ca) nên mỗi 1 TP/FP làm precision/recall nhảy 20–25%, và các lỗi `WRONG_CLASS` trong calib chủ yếu là nhãn người bị đổi thành `person`/`car` trong khi trên `raw` lại nghiêng về nhầm giữa `car`/`truck`/`bus`. Trên `raw`, với `phash_dist=8` bước 1 loại **16** ảnh trùng (`8` exact + `3` near + `5` leak test, thêm `3` `BAD_SIZE` và `2` `CORRUPT_IMAGE` = `21/224`), tổng số loại hợp lý; nhưng vì `second_opinion` trên `raw` vẫn sinh tới **128** cảnh báo, ngân sách 80 ảnh không đủ phủ hết.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Chứng minh cửa chặn hoạt động:**
Khi chạy với ngân sách người duyệt bằng 0:
`python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`
Vì không có người gán 6 ảnh `MISSING_LABEL_FILE`, vòng 3 còn **6 lỗi bắt buộc (`ERROR`)**, pipeline kết luận **REWORK**, trả về **mã thoát 20**, in `"KHÔNG xuất: còn lỗi bắt buộc (Rework)"` và **không tạo thư mục `work/exp_nb`**.

**(b) Các điểm mù pipeline KHÔNG bắt được (≥ 2 điểm):**
1. **Ảnh trùng bị cắt/thu nhỏ hoặc biến đổi mạnh:** Trong `calib`, `DUP_IMAGE_NEAR` ở mức biến đổi mạnh vẫn bị sót (recall cao nhất chỉ 0.67 ở `phash_dist=8`), vì pHash thay đổi lớn khi crop sâu.
2. **Box lỏng nhẹ (`0.5 <= IoU < 0.7`) hoặc nhầm lớp mà model `yolo26l` cũng nhầm:** `second_opinion` chỉ gắn cờ `POSSIBLE_LOOSE_BOX` khi `IoU < 0.5` và chỉ dựa vào `yolo26l` làm tham chiếu, nên nếu cả nhãn gốc lẫn `yolo26l` cùng nhầm lớp (ví dụ `car` vs `truck` ở góc khuất) hoặc vật có `conf < 0.6`, pipeline sẽ im lặng.
3. **45 ảnh cuối hàng đợi không được duyệt:** Với ngân sách 80/125 ảnh, các cảnh báo có trọng số thấp hơn (như `BOX_TINY`, `EMPTY_LABEL`, `POSSIBLE_LOOSE_BOX`) ở cuối hàng đợi không qua mắt người nếu không tăng ngân sách duyệt.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
| biến thể | #ảnh train | số seed | map50 (TB ± độ lệch) | map50_95 (TB ± độ lệch) | precision (TB ± độ lệch) | recall (TB ± độ lệch) |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 203 | 2 | 0.699 ± 0.025 | 0.486 ± 0.013 | 0.796 ± 0.050 | 0.596 ± 0.049 |

## 6. Quyết định và ghi chú AI
**(a) Quyết định cuối:** Giữ nguyên kết luận **HOLD** (chưa phát hành chính thức). Để chuyển sang **Release**, cần bổ sung ngân sách duyệt người (từ 80 lên ~125 ảnh hoặc thêm 1 vòng duyệt) để xử lý nốt **38** ảnh còn cảnh báo, đưa tỉ lệ cảnh báo chưa duyệt xuống dưới ngưỡng **10.00%**.

**(b) Ghi chú dùng AI:**
- **Prompt chính:** Cung cấp đặc tả 4 module (`label_format`, `gate`, `autofix`, `second_opinion`), cấu trúc `Issue`/`Dataset` trong `common.py`, các mã lỗi/ngưỡng trong `configs/pipeline.yaml` và yêu cầu xử lý các ca biên (dòng trống, nhiều lỗi trên 1 dòng, ảnh đã xoá).
- **Cách kiểm:** Chạy toàn bộ `pytest -q` (79/79 test xanh gồm cả e2e), chạy `pipeline.calibrate` và `pipeline.run` (kể cả `--budget 0` kiểm tra mã thoát 20), đối chiếu `02_error_list_v1.txt`, `03_fixes.csv`, `07_error_list_v3.txt` và `08_summary.json`.
- **1 lần AI sai:** Khi chạy lệnh lần đầu trên máy Windows, `python` mặc định trỏ vào Python 3.14 chưa cài `PyYAML` trong khi thư viện cài ở Python 3.12, cần chỉ định đúng trình thông dịch Python 3.12 (hoặc `.venv`) để chạy test và pipeline.
