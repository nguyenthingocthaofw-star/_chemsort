# Reports

Thư mục `reports/` lưu kết quả đánh giá, export TFLite và so sánh model của
ChemSort. Đây là nơi truy vết quyết định chọn model, không chứa model binary.

## Kết luận hiện tại

Model triển khai trong app là **YOLO11n-cls** (dataset v2), file
`assets/chemsort/yolo11n-cls_data_v2_float32.tflite`. Đánh giá lại chính file
TFLite đó bằng pipeline mới (tiền xử lý giống hệt app):

| Split | Ảnh | Acc 6 | Macro-P 6 | Macro-R 6 | **Macro-F1 6** | Acc 9 | Macro-F1 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 713 | 0.9481 | 0.9510 | 0.9458 | 0.9482 | 0.9341 | 0.9328 |
| test | 713 | 0.9215 | 0.9179 | 0.9184 | **0.9171** | 0.9088 | 0.9127 |
| test_ext | 594 | 0.9327 | 0.9270 | 0.9260 | **0.9260** | 0.9276 | 0.8183\* |

\* test_ext không có Vegetation; macro 9 lớp tính cả lớp vắng mặt (0 điểm).

So với pipeline v1 trên split test (`comparison/test/`). Lưu ý: final-test v1
được chạy trên một bản manifest cũ hơn, 30 ảnh Food Organics khác với test hiện
tại (683/713 ảnh trùng), nên chênh lệch chỉ mang tính tham khảo; kiểm định
McNemar trên 683 ảnh chung cho p < 0,001 với cả hai model v1.

| Model | Train data | Macro-F1 6 [95% CI] | Acc 6 |
| --- | --- | ---: | ---: |
| YOLO11n-cls TFLite float32 (app) | v2 | 0.9171 [0.894, 0.937] | 0.9215 |
| EfficientNet-Lite0 TFLite float16 | v1 | 0.8500 [0.821, 0.877] | 0.8583 |
| MobileNetV2 fine-tuned (Keras) | v1 | 0.8304 [0.800, 0.858] | 0.8401 |

### Benchmark trên dữ liệu augment (`v2_aug_medium`)

Train trên `data/augmented/v2_medium/manifest_v2_aug_medium.csv` (train cân bằng
1.648 ảnh/lớp bằng augment offline preset `medium`, cộng augment online
`medium`); val/test/test_ext giữ nguyên nên so trực tiếp được với model app.
YOLO train trên GPU (RTX 3060 Laptop), Keras train trên CPU. Model Keras được
đánh giá ở dạng TFLite float16 (parity 100% với Keras).

| Model | Size MB | Macro-F1 6 val | **Macro-F1 6 test** | Macro-F1 6 test_ext | Macro-F1 9 test |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n-cls v2 (app, TFLite) | 5.94 | **0.9482** | 0.9171 | **0.9260** | 0.9127 |
| YOLO11n-cls v2_aug_medium (PyTorch) | 3.06 | 0.9374 | **0.9258** | 0.9187 | 0.9136 |
| MobileNetV3Large v2_aug_medium (TFLite fp16) | 5.79 | 0.8464 | 0.8679 | 0.9115 | 0.8478 |
| EfficientNetB0 v2_aug_medium (TFLite fp16) | 7.78 | 0.8590 | 0.8484 | 0.8949 | 0.8360 |
| MobileNetV2 v2_aug_medium (TFLite fp16) | 4.33 | 0.8117 | 0.8128 | 0.9037 | 0.8031 |

McNemar (6 nhóm, test): YOLO app vs YOLO aug p = 0.51 (không khác biệt có ý
nghĩa); YOLO app vs MobileNetV3Large p = 2.6e-4, vs EfficientNetB0 p = 1.8e-6,
vs MobileNetV2 p = 5.9e-11; MobileNetV3Large vs EfficientNetB0 p = 0.18.

Kết luận: YOLO11n-cls vẫn tốt nhất và giữ nguyên model trong app; augment
offline không cải thiện YOLO có ý nghĩa thống kê. Trong nhóm Keras,
MobileNetV3Large tốt nhất trên test/test_ext. MobileNetV3Large và
EfficientNetB0 đạt val loss tốt nhất ở epoch cuối, nên train thêm
(`--finetune-epochs`) có thể còn cải thiện.

| Câu hỏi | File/thư mục |
| --- | --- |
| Model nào tốt nhất trên test / test_ext? | `comparison/<split>/comparison.md` |
| Confusion matrix của các model cạnh nhau? | `comparison/<split>/confusion_matrices_6class.png`, `..._9class.png` |
| F1 từng lớp của từng model? | `comparison/<split>/per_class_f1_6class.png` |
| Hai model khác nhau có ý nghĩa thống kê không? | `comparison/<split>/mcnemar_6class.csv` |
| Chi tiết một model trên một split? | `experiments/<run>/<artifact>/<split>/` |
| Export TFLite có giữ đúng dự đoán không? | `experiments/<run>/export_report.json` |

## Báo cáo pipeline mới

```text
reports/
  experiments/<run>/
    history.csv, training_curves.png      lịch sử train (val loss/accuracy)
    export_report.json                    kích thước, SHA-256, parity TFLite
    <artifact>/<split>/                   artifact = pytorch | keras | tflite_float32 | tflite_float16 | tflite
      metrics.json, predictions.csv
      report_9class.txt, report_6class.txt
      confusion_matrix_9class.png, confusion_matrix_6class.png
      per_class_9class.png, per_class_6class.png, reliability_6class.png
  comparison/<split>/                     tạo bởi pipelines.compare_models
```

`experiments/legacy_v1_*` được tạo lại từ `final_test/*_predictions.csv` bằng
`python -m tools.import_legacy_reports`.

---

# Báo cáo pipeline v1 (legacy)

Các thư mục `data/`, `validation/`, `tflite/`, `final_test/` dưới đây do
`pipelines/legacy/` tạo ra trên dataset v1 (chỉ RealWaste). Giữ nguyên để truy
vết; không chỉnh tay.

## Cấu trúc thư mục

```text
reports/
  README.md
  data/
    data_batch.png
    report_v0.txt
  validation/
    model_validation_comparison.csv
    efficientnet_lite0_no_aug/
    mobilenetv2_aug/
    mobilenetv2_no_aug/
    mobilenetv2_no_aug_finetuned/
    mobilenetv3large_no_aug/
    mobilenetv3large_no_aug_finetuned/
    mobilenetv3large_no_aug_finetuned_extended/
  tflite/
    tflite_candidate_comparison.csv
    efficientnet_lite0_tflite_export_report.json
    mobilenetv2_tflite_export_report.json
    mobilenetv3large_tflite_export_report.json
  final_test/
    efficientnet_lite0_float16_tflite_test/
    mobilenetv2_no_aug_finetuned_test/
```

## Data reports

`reports/data/` được tạo bởi `pipelines/legacy/check_data_loader.py`.

| File | Ý nghĩa |
| --- | --- |
| `data_batch.png` | Ảnh minh họa một batch sau khi data loader đọc, resize và batch dữ liệu. |
| `report_v0.txt` | Thứ tự class, shape batch, dtype và khoảng pixel. |

Thông tin hiện tại:

- Image shape: `(32, 224, 224, 3)`
- Label shape: `(32,)`
- Image dtype: `float32`
- Pixel range: `0.0` đến `255.0`
- Thứ tự class: `Cardboard`, `Food Organics`, `Glass`, `Metal`,
  `Miscellaneous Trash`, `Paper`, `Plastic`, `Textile Trash`, `Vegetation`

## Validation reports

`reports/validation/` chứa kết quả trên split `val`. Các run validation không
dùng test split; trường `test_set_used` trong các report hiện đều là `false`.

Bảng tổng hợp chính:

```text
reports/validation/model_validation_comparison.csv
```

Kết quả validation hiện tại, sắp xếp theo `macro_f1_6class`:

| Run | Stage | Augmentation | Acc 9 lớp | Macro-F1 9 lớp | Acc 6 lớp | Macro-F1 6 lớp |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `efficientnet_lite0_no_aug` | baseline | off | 0.8527 | 0.8515 | 0.8892 | 0.8861 |
| `mobilenetv3large_no_aug_finetuned` | finetune | off | 0.8429 | 0.8450 | 0.8766 | 0.8774 |
| `mobilenetv3large_no_aug_finetuned_extended` | finetune continued | off | 0.8415 | 0.8423 | 0.8766 | 0.8759 |
| `mobilenetv2_no_aug_finetuned` | finetuned | off | 0.8303 | 0.8299 | 0.8682 | 0.8684 |
| `mobilenetv3large_no_aug` | baseline | off | 0.8373 | 0.8405 | 0.8654 | 0.8640 |
| `mobilenetv2_no_aug` | frozen backbone baseline | off | 0.7938 | 0.7920 | 0.8429 | 0.8424 |
| `mobilenetv2_aug` | frozen backbone baseline | on | 0.7798 | 0.7740 | 0.8317 | 0.8242 |

Một thư mục run validation thường có:

| Pattern | Ý nghĩa |
| --- | --- |
| `*_metrics.json` | Metric chính, cấu hình run, seed, epoch, model path và `test_set_used`. |
| `*_comparison.json` | So sánh baseline và fine-tuned, hiện dùng cho MobileNetV2 fine-tune. |
| `*_history.csv` | Lịch sử train/validation loss và accuracy theo epoch. |
| `*_training_curves.png` | Đồ thị training curve. |
| `*_9class_val_report.txt` | Classification report cho 9 lớp RealWaste trên validation. |
| `*_6class_val_report.txt` | Classification report sau khi gom thành 6 nhóm ChemSort. |
| `*_9class_val_confusion_matrix.png` | Confusion matrix 9 lớp. |
| `*_6class_val_confusion_matrix.png` | Confusion matrix 6 lớp. |

## TFLite reports

`reports/tflite/` ghi kết quả export model sang TensorFlow Lite và kiểm parity
giữa Keras/reference model với TFLite. Các export report không dùng test split.

```text
reports/tflite/tflite_candidate_comparison.csv
```

| Candidate | TFLite khuyến nghị | Size | Val macro-F1 6 lớp | Agreement 9 lớp | Agreement 6 lớp | Max probability diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EfficientNet-Lite0 | float16 | 6.4686 MB | 0.8861 | 1.0000 | 1.0000 | 0.007282 |
| MobileNetV3Large | float32 | 11.4083 MB | 0.8774 | 1.0000 | 1.0000 | 0.000004 |
| MobileNetV2 | float32 | 8.5226 MB | 0.8684 | 1.0000 | 1.0000 | 0.000009 |

Mỗi `*_tflite_export_report.json` thường có:

- source model/weights và SHA-256.
- full validation metrics của model nguồn.
- subset parity: 90 ảnh validation, 10 ảnh cho mỗi lớp gốc.
- thông tin input/output tensor TFLite.
- metric và parity cho float32, float16.
- `recommended_mobile_model` và `recommended_format`.

## Final-test reports

`reports/final_test/` chứa đánh giá cuối cùng trên split `test`. Đây là báo cáo
một lần sau khi đã chọn model; không dùng kết quả ở đây để tune model.

| Run | Artifact | Test images | Acc 9 lớp | Macro-F1 9 lớp | Acc 6 lớp | Macro-F1 6 lớp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `efficientnet_lite0_float16_tflite_test` | `efficientnet_lite0_float16.tflite` | 713 | 0.8247 | 0.8298 | 0.8583 | 0.8500 |
| `mobilenetv2_no_aug_finetuned_test` | `mobilenetv2_no_aug_finetuned.keras` | 713 | 0.8205 | 0.8310 | 0.8401 | 0.8304 |

Một thư mục final test có:

| Pattern | Ý nghĩa |
| --- | --- |
| `*_summary.json` | Metric chính, confidence interval 95%, SHA model, test count, test-vs-validation gap và cấu hình inference. |
| `*_predictions.csv` | Dự đoán từng ảnh test, gồm nhãn thật, nhãn dự đoán và xác suất. |
| `*_9class_report.txt` | Classification report 9 lớp trên test. |
| `*_6class_report.txt` | Classification report 6 nhóm ChemSort trên test. |
| `*_9class_confusion_matrix.png` | Confusion matrix 9 lớp trên test. |
| `*_6class_confusion_matrix.png` | Confusion matrix 6 lớp trên test. |

## Ý nghĩa metric

| Metric | Ý nghĩa |
| --- | --- |
| `accuracy_9class` | Tỷ lệ dự đoán đúng trên 9 nhãn gốc RealWaste. |
| `macro_f1_9class` | F1 trung bình đều theo 9 lớp, hữu ích khi lớp mất cân bằng. |
| `accuracy_6class` | Tỷ lệ đúng sau khi gom 9 output thành 6 nhóm ChemSort. |
| `macro_f1_6class` | F1 trung bình đều theo 6 nhóm ChemSort; đây là metric ưu tiên khi chọn model cho app. |
| `organic_recall_6class` | Khả năng bắt đúng rác hữu cơ. |
| `plastic_precision_6class` | Độ chính xác khi model dự đoán nhựa. |
| `plastic_recall_6class` | Khả năng bắt đúng nhựa. |
| `agreement_9class` / `agreement_6class` | Tỷ lệ TFLite cho cùng nhãn dự đoán với model reference trong kiểm tra export. |

## Quy tắc bảo trì

- Không chỉnh tay report sau khi đã dùng làm căn cứ so sánh.
- Không dùng `final_test/` để chọn model hoặc điều chỉnh tham số.
- Nếu chạy lại training/export, hãy để script tạo lại report tương ứng thay vì
  sửa thủ công.
- Nếu `data/manifest.csv` thay đổi, toàn bộ validation, TFLite export và final
  test liên quan cần được tạo lại.
- Khi chọn artifact mới cho Flutter, cập nhật đồng bộ `models/README.md`,
  `pipelines/README.md`, `reports/tflite/tflite_candidate_comparison.csv` và final-test
  summary.
