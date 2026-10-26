# Pipelines

CLI orchestration cho pipeline ML. Chạy từ `ml/` bằng
`python -m pipelines.<module>`; mỗi module có `--help`. Logic dùng chung
(contract, metrics, tiền xử lý, báo cáo) nằm trong `src/chemsort/`, không đặt
trong các module này.

| Module | Vai trò | Môi trường |
| --- | --- | --- |
| `train_yolo` | Train Ultralytics YOLO-cls (`yolo11n-cls`, `yolov8n-cls`, …); `--augment-preset` | YOLO |
| `train_keras` | Train MobileNetV2/V3Small/V3Large, EfficientNetB0/V2B0/Lite0; `--augment-preset`, `--mixup`, `--cutmix` | TF |
| `export_tflite` | Export TFLite float32/float16, kiểm tra parity, ghi mapping | theo run |
| `evaluate` | Đánh giá checkpoint / TFLite trên val, test, test_ext | theo artifact |
| `compare_models` | Bảng xếp hạng, biểu đồ, confusion matrix, McNemar | chung |
| `run_experiments` | Chạy chuỗi train → export → evaluate → compare | chung |
| `predict_tflite` | Phân loại một ảnh bằng TFLite | TFLite |

`legacy/` giữ nguyên pipeline TensorFlow v1 (RealWaste-only) đã tạo các báo
cáo trong `reports/validation`, `reports/tflite`, `reports/final_test`:

| Module | Vai trò |
| --- | --- |
| `legacy.check_data_loader` | Kiểm tra manifest v1, sinh ảnh batch mẫu |
| `legacy.train_candidate` | Train EfficientNet-Lite0 / MobileNetV3Large |
| `legacy.train_mobilenetv2`, `legacy.finetune_mobilenetv2` | MobileNetV2 baseline + fine-tune |
| `legacy.export_candidate_tflite`, `legacy.export_mobilenetv2_tflite` | Export v1 |
| `legacy.evaluate_efficientnet_lite0_tflite_test`, `legacy.evaluate_mobilenetv2_test` | Final test v1 |

Chuẩn bị dữ liệu nằm trong `tools/` (`dataset_stats`, `preview_augmentation`,
`augment_dataset`, `package_data`); mọi pipeline nhận manifest đã augment qua
`--manifest data/augmented/<tag>/manifest_<...>.csv`.

Luồng đầy đủ và các lệnh mẫu nằm trong [`../README.md`](../README.md).
