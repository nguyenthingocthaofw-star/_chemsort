# ChemSort ML — train, đánh giá và so sánh model

Thư mục này chứa toàn bộ phần machine learning của ChemSort: chuẩn bị dữ liệu,
train nhiều kiến trúc (YOLO-cls, MobileNet, EfficientNet), export TFLite,
đánh giá thống nhất bằng **confusion matrix + precision / recall / F1 / …** và
so sánh các model với nhau. Model cuối cùng được chép sang app Flutter ở thư
mục cha (`../assets/chemsort/`).

Mọi lệnh dưới đây chạy từ thư mục `ml/`.

## Mục lục

1. [Cấu trúc](#1-cấu-trúc)
2. [Cài đặt](#2-cài-đặt)
3. [Dữ liệu](#3-dữ-liệu)
4. [Train](#4-train)
5. [Export TFLite](#5-export-tflite)
6. [Đánh giá](#6-đánh-giá)
7. [So sánh model](#7-so-sánh-model)
8. [Chạy trọn bộ benchmark](#8-chạy-trọn-bộ-benchmark)
9. [Đưa model vào app](#9-đưa-model-vào-app)
10. [Quy tắc](#10-quy-tắc)

## 1. Cấu trúc

```text
ml/
  src/chemsort/        Thư viện dùng chung
    contracts.py         Thứ tự 9 lớp + quy tắc gộp 9 → 6 nhóm (nguồn chuẩn duy nhất)
    data.py              Đọc manifest v1/v2, chuẩn hóa cột, chọn split
    preprocess.py        Tiền xử lý giống hệt app (crop vuông giữa + area resize)
    augment.py           Preset augmentation dùng chung (offline PIL, Keras, YOLO, MixUp/CutMix)
    metrics.py           Accuracy, precision, recall, F1, kappa, MCC, ECE, CI, McNemar
    plots.py             Confusion matrix, biểu đồ theo lớp, so sánh nhiều model
    evaluation.py        Ghi báo cáo đánh giá đầy đủ cho một model trên một split
    predictors.py        Chạy Keras / PyTorch-YOLO / TFLite với cùng tensor đầu vào
    keras_models.py      Model zoo Keras (MobileNetV2/V3, EfficientNetB0/V2B0/Lite0)
    yolo_data.py         Dựng cây thư mục split/lớp/ảnh cho Ultralytics
    inference.py         Phân loại một ảnh bằng TFLite
  pipelines/           Các lệnh chính
    train_keras.py       Train model Keras (2 giai đoạn: head → fine-tune)
    train_yolo.py        Train YOLO-cls bằng Ultralytics
    export_tflite.py     Export TFLite float32/float16 + kiểm tra parity
    evaluate.py          Đánh giá một artifact trên val / test / test_ext
    compare_models.py    Bảng xếp hạng + biểu đồ so sánh nhiều model
    run_experiments.py   Chạy trọn chuỗi train → export → evaluate → compare
    predict_tflite.py    Dự đoán một ảnh
    legacy/              Pipeline TensorFlow v1 ban đầu (giữ để tái lập báo cáo cũ)
  tools/
    extract_data_bundle.py     Giải nén ChemSort_Data.zip vào data/
    prepare_data.py            Tạo manifest v1 (RealWaste) — đã khóa
    build_dataset_v2.py        Tạo manifest v2 (thêm 3 nguồn ảnh một vật thể)
    dataset_stats.py           Thống kê dataset: phân bố lớp/split/nguồn, kích thước, màu
    preview_augmentation.py    Lưới ảnh trước/sau augment cho từng preset
    augment_dataset.py         Sinh ảnh augment offline (cân bằng lớp) + manifest mới
    package_data.py            Đóng gói data/raw và data/augmented thành file zip
    import_legacy_reports.py   Đưa kết quả final-test v1 vào định dạng báo cáo mới
    sync_flutter_assets.py     Chép model đã export vào app Flutter
    validate_chemsort_assets.py  Kiểm tra model/labels/mapping/flashcards của app
  data/                manifest.csv (v1), manifest_v2.csv (v2), thống kê, phiên bản
    augmented/<tag>/     Ảnh augment + manifest (sinh bởi tools.augment_dataset, gitignore)
  models/<run>/        Checkpoint, train_config.json, tflite/ (binary bị gitignore)
  reports/
    experiments/<run>/<artifact>/<split>/   Báo cáo đánh giá từng model
    comparison/<split>/                     So sánh nhiều model
    validation/, tflite/, final_test/, data/  Báo cáo của pipeline v1 (legacy)
  tests/               Unit test (chạy không cần TensorFlow/PyTorch)
```

## 2. Cài đặt

Python 3.11 hoặc 3.12. Nên dùng **hai môi trường riêng** vì TensorFlow và
PyTorch hay xung đột phiên bản:

```powershell
# Môi trường YOLO (PyTorch + Ultralytics)
python -m venv .venv-yolo
.\.venv-yolo\Scripts\Activate.ps1
python -m pip install -r requirements-yolo.txt

# Môi trường Keras (TensorFlow)
python -m venv .venv-tf
.\.venv-tf\Scripts\Activate.ps1
python -m pip install -r requirements-tf.txt
```

Có GPU NVIDIA thì cài bản `torch` CUDA trước (xem https://pytorch.org) rồi mới
cài `requirements-yolo.txt`.

Kiểm tra nhanh (chỉ cần `requirements.txt`):

```powershell
python -m unittest discover -s tests -v
```

## 3. Dữ liệu

### 3.1 Giải nén bộ dữ liệu của nhóm

Đặt `ChemSort_Data.zip` ở thư mục gốc repo (cạnh `pubspec.yaml`), rồi:

```powershell
python -m tools.extract_data_bundle                      # ảnh gốc → data/raw/
python -m tools.extract_data_bundle --parts processed    # (tùy chọn) cây Ultralytics có sẵn
```

Kết quả:

```text
data/raw/RealWaste/                       9 lớp, 4.752 ảnh
data/raw/TrashNet/                        6 lớp
data/raw/DrinkingWasteClassification/     rawimgs/{AluCan,Glass,HDPEM,PET}
data/raw/Garbage_Classification/          12 lớp
```

Không có file zip thì tải từng nguồn (xem README gốc, mục 10) và giải nén
đúng cấu trúc trên.

### 3.2 Manifest

| File | Nội dung | Split |
| --- | --- | --- |
| `data/manifest.csv` | v1 — chỉ RealWaste | train 3.326 · val 713 · test 713 |
| `data/manifest_v2.csv` | v2 — RealWaste + 3 nguồn bổ sung | train 8.717 · val 713 · test 713 · test_ext 594 |

Mỗi dòng: đường dẫn ảnh (tương đối với `data/raw`), nhãn 9 lớp, split, nguồn,
nhãn gốc của nguồn, kích thước, SHA-256, pHash. Val/test của v1 và v2 **giống
hệt nhau**, nên mọi model đều so được trên cùng một tập test.

Hai manifest đã khóa; mọi con số báo cáo đều dựa trên chúng. Muốn tạo phiên
bản dữ liệu mới:

```powershell
python -m tools.build_dataset_v2                     # ghi data/manifest_v2.rebuilt.csv và so với bản khóa
python -m tools.build_dataset_v2 --cap 600 --output data/manifest_v3.csv
```

Các pipeline nhận `--manifest v1`, `--manifest v2` (mặc định) hoặc đường dẫn CSV.

### 3.3 Thống kê dữ liệu

```powershell
python -m tools.dataset_stats                  # → reports/data/v2/
python -m tools.dataset_stats --manifest v1    # → reports/data/v1/
```

Ghi số ảnh theo lớp × split (9 lớp và 6 nhóm), theo nguồn, tỉ lệ mất cân bằng,
kích thước/tỉ lệ khung ảnh, màu trung bình theo lớp (`class_distribution.png`,
`source_distribution.png`, `image_sizes.png`, `pixel_stats.png`, `summary.json`).

### 3.4 Augmentation

Mọi chính sách augment nằm ở `src/chemsort/augment.py`, gồm 4 preset:

| Preset | Hình học | Màu sắc | Khác |
| --- | --- | --- | --- |
| `none` | — | — | — |
| `light` | lật ngang, xoay ±10°, zoom ≤10%, dịch ≤5% | sáng/tương phản/bão hòa ±10%, hue ±1% | — |
| `medium` (mặc định) | lật ngang, xoay ±20°, zoom ≤15%, dịch ≤8% | sáng/tương phản ±20%, bão hòa ±30%, hue ±1,5% | blur 10%, nhiễu σ≤4 |
| `strong` | + lật dọc 20%, xoay ±30°, zoom ≤25%, dịch ≤12% | ±30%, bão hòa ±50%, hue ±3% | blur 20%, nhiễu σ≤8, random erasing 40% |

Sau khi xoay, ảnh được cắt theo hình chữ nhật lớn nhất nằm trọn trong vùng ảnh
thật, nên **không có góc đen**. Augment chỉ áp dụng cho **train**; val/test/test_ext
không bao giờ bị biến đổi.

Hai cách dùng:

1. **Online** (mỗi epoch một biến thể ngẫu nhiên, không tốn ổ đĩa):
   `train_keras --augment-preset strong --mixup 0.2 --cutmix 1.0`,
   `train_yolo --augment-preset light` (YOLO dùng các tham số tương ứng của
   Ultralytics; `medium` = mặc định Ultralytics, cấu hình của model trong app).
2. **Offline** (sinh ảnh mới để cân bằng lớp, ví dụ Vegetation 305 → 1.648):

```powershell
python -m tools.preview_augmentation                 # xem trước: reports/data/augmentation/preview_<preset>.png
python -m tools.augment_dataset                      # balance, preset medium → data/augmented/v2_medium/
python -m tools.augment_dataset --preset strong --target-per-class 2000
python -m tools.augment_dataset --mode multiply --copies 1 --preset light
python -m tools.dataset_stats --manifest data/augmented/v2_medium/manifest_v2_aug_medium.csv

# Train trên dữ liệu đã augment (tên run: <model>_v2_aug_medium)
python -m pipelines.train_keras --model efficientnetb0 --manifest data/augmented/v2_medium/manifest_v2_aug_medium.csv
python -m pipelines.train_yolo --manifest data/augmented/v2_medium/manifest_v2_aug_medium.csv
```

`augment_dataset` ghi `images/<lớp>/*.jpg`, manifest (dòng gốc `base=raw` +
dòng sinh ra `base=data`, `source=augmented`, có `parent_path` và `aug_seed`
để truy vết), `augmentation_report.json`, `class_counts.png` và `examples.png`.
Mỗi ảnh có seed riêng suy ra từ `--seed` và hash ảnh gốc, nên chạy lại cho
đúng kết quả cũ. Bộ nạp manifest từ chối mọi ảnh augment nằm ngoài split train.

Lưu ý khi đã augment offline: tắt bớt augment online (`--augment-preset light`)
để tránh biến đổi chồng lên nhau quá mạnh.

### 3.5 Đóng gói dữ liệu

```powershell
python -m tools.package_data                         # → ../ChemSort_Data_raw.zip, ../ChemSort_Data_augmented_v2_medium.zip
```

Đường dẫn trong zip tính từ `ml/`, nên giải nén trong `ml/` là khôi phục đúng
cấu trúc. Manifest augment tham chiếu ảnh gốc trong `data/raw`, nên cần cả hai zip.

## 4. Train

Tên run mặc định là `<model>_<dataset>`, ví dụ `yolo11n-cls_v2`,
`mobilenetv3large_v2`. Sau khi train, mỗi lệnh tự đánh giá model trên **val**
bằng đúng bộ tiền xử lý của app.

### 4.1 YOLO-cls (Ultralytics)

```powershell
python -m pipelines.train_yolo                          # YOLO11n-cls, cấu hình của model trong app
python -m pipelines.train_yolo --model yolov8n-cls
python -m pipelines.train_yolo --model yolo11s-cls --batch 32 --device 0
```

Mặc định: 60 epoch, batch 64, ảnh 224, patience 15, `--augment-preset medium`
(RandAugment, random erasing 0,4, lật ngang 0,5, HSV jitter), seed 42,
deterministic. Từng tham số ghi đè được: `--erasing`, `--fliplr`, `--flipud`,
`--hsv-h/s/v`, `--auto-augment none`. Cây thư mục
`data/build/yolo_v2/<split>/<lớp>/<nguồn>_<sha16>.jpg` được dựng bằng hard
link (không tốn thêm dung lượng); có thể trỏ thẳng vào cây đã giải nén bằng
`--dataset-dir data/processed/dataset`.

### 4.2 Keras (MobileNet, EfficientNet)

```powershell
python -m pipelines.train_keras --model mobilenetv3large
python -m pipelines.train_keras --model mobilenetv2
python -m pipelines.train_keras --model mobilenetv3small
python -m pipelines.train_keras --model efficientnetb0
python -m pipelines.train_keras --model efficientnetv2b0
python -m pipelines.train_keras --model efficientnet_lite0 --finetune-epochs 0
```

Hai giai đoạn trong một lệnh: (1) đóng băng backbone, train head
(`--epochs 15`, lr 1e-3); (2) mở 30 layer cuối, BatchNorm vẫn đóng băng
(`--finetune-epochs 10`, lr 1e-5). Giữ trọng số có val loss thấp nhất.
EfficientNet-Lite0 là module TF Hub cũ nên chỉ train được head.

Tùy chọn hay dùng: `--augment-preset none|light|medium|strong`,
`--mixup 0.2`, `--cutmix 1.0` (trộn cặp ảnh, loss chuyển sang one-hot),
`--class-weight`, `--batch-size`, `--limit-per-class 20` (chạy thử nhanh).

Kết quả của một run:

```text
models/<run>/train_config.json      cấu hình, siêu tham số, input contract, số tham số
models/<run>/best.pt | best.weights.h5 (+ best.keras)
reports/experiments/<run>/history.csv, training_curves.png
reports/experiments/<run>/<pytorch|keras>/val/   báo cáo đánh giá trên val
```

## 5. Export TFLite

```powershell
python -m pipelines.export_tflite --run yolo11n-cls_v2
python -m pipelines.export_tflite --run mobilenetv3large_v2 --formats float16
```

Tạo `models/<run>/tflite/<run>_float32.tflite`, `<run>_float16.tflite`,
`labels_9.txt`, `chemsort_mapping.json` (có khối `input` mô tả đúng layout,
khoảng pixel và cách resize) và `reports/experiments/<run>/export_report.json`.
Parity được đo trên 90 ảnh val (10 ảnh/lớp): format chỉ **PASS** khi trùng dự
đoán với model gốc ≥ 98% (9 lớp) và ≥ 99% (6 nhóm).

## 6. Đánh giá

```powershell
# Model đã train (checkpoint gốc hoặc bản TFLite)
python -m pipelines.evaluate --run yolo11n-cls_v2 --split val
python -m pipelines.evaluate --run yolo11n-cls_v2 --artifact tflite_float32 --split test test_ext

# Bất kỳ file TFLite nào, ví dụ model đang nằm trong app
python -m pipelines.evaluate --tflite ../assets/chemsort/yolo11n-cls_data_v2_float32.tflite `
    --mapping ../assets/chemsort/chemsort_mapping.json --name app_yolo11n-cls_v2 --split test test_ext
```

Mọi model được đưa vào **cùng một tensor** do `preprocess.py` tạo ra (port
từng bước của `preprocessImage` trong `lib/services/yolo_classifier.dart`), nên
điểm số là điểm điện thoại thực sự đạt được.

Mỗi lần đánh giá ghi vào `reports/experiments/<run>/<artifact>/<split>/`:

| File | Nội dung |
| --- | --- |
| `metrics.json` | Toàn bộ metric, khoảng tin cậy 95%, SHA-256 model, thời gian suy luận |
| `predictions.csv` | Từng ảnh: nhãn thật, dự đoán, độ tin cậy, xác suất 9 lớp + 6 nhóm |
| `report_9class.txt`, `report_6class.txt` | Precision / recall / F1 / specificity / support từng lớp |
| `confusion_matrix_9class.png`, `confusion_matrix_6class.png` | Số lượng + % theo hàng (recall) |
| `per_class_9class.png`, `per_class_6class.png` | Cột precision / recall / F1 từng lớp |
| `reliability_6class.png` | Đường calibration (độ tin cậy so với độ chính xác) |

### Ý nghĩa metric

| Metric | Ý nghĩa |
| --- | --- |
| Accuracy | Tỷ lệ ảnh đoán đúng. |
| Precision (lớp c) | Trong các ảnh model gọi là c, bao nhiêu % đúng là c. |
| Recall (lớp c) | Trong các ảnh thật sự là c, model bắt được bao nhiêu %. |
| F1 | Trung bình điều hòa của precision và recall. |
| Macro (P/R/F1) | Trung bình đều theo lớp — mỗi lớp nặng như nhau dù ít ảnh. **Macro-F1 6 nhóm là metric chính để chọn model cho app.** |
| Weighted F1 | Trung bình theo số ảnh mỗi lớp. |
| Balanced accuracy | Trung bình recall các lớp có mặt. |
| Specificity | Trong các ảnh *không* phải c, bao nhiêu % model không gọi nhầm là c. |
| Cohen's kappa, MCC | Độ đồng thuận đã trừ phần đoán may; 1 = hoàn hảo, 0 = ngẫu nhiên. |
| Top-3 accuracy | Nhãn đúng nằm trong 3 lớp có xác suất cao nhất (9 lớp). |
| Log loss, ECE | Chất lượng xác suất; ECE thấp = độ tin cậy phản ánh đúng khả năng đúng (quan trọng vì app dùng ngưỡng 0,5). |
| 95% CI | Khoảng tin cậy bootstrap (2.000 lần lấy mẫu lại). |

Ở mức **6 nhóm**, xác suất các lớp trong cùng nhóm được cộng lại trước khi
lấy argmax — đúng quy tắc của `mapping_service.dart`.

`test_ext` không có lớp Vegetation, nên macro 9 lớp bị kéo xuống bởi một lớp
0 điểm; báo cáo ghi thêm `macro_f1_present` (chỉ tính lớp có mặt).

## 7. So sánh model

```powershell
python -m pipelines.compare_models --split test
python -m pipelines.compare_models --split test_ext
python -m pipelines.compare_models --split val --runs yolo mobilenet
```

Ghi vào `reports/comparison/<split>/`:

| File | Nội dung |
| --- | --- |
| `comparison.md`, `comparison.csv` | Bảng xếp hạng theo macro-F1 6 nhóm, kèm CI, kích thước, tốc độ |
| `metrics_6class.png`, `metrics_9class.png` | Cột accuracy, precision, recall, F1, kappa, MCC của mọi model (có thanh CI) |
| `per_class_f1_6class.png`, `per_class_f1_9class.png` | Heatmap F1 model × lớp (ô đậm = tốt nhất) |
| `confusion_matrices_6class.png`, `confusion_matrices_9class.png` | Confusion matrix các model đặt cạnh nhau |
| `mcnemar_6class.csv/.png`, `mcnemar_9class.csv` | Kiểm định McNemar từng cặp: p < 0,05 nghĩa là hai model khác nhau có ý nghĩa thống kê |
| `efficiency.png` | Macro-F1 theo kích thước file |

Kết quả final-test của pipeline v1 (EfficientNet-Lite0, MobileNetV2) được đưa
vào cùng bảng bằng `python -m tools.import_legacy_reports`.

## 8. Chạy trọn bộ benchmark

```powershell
# Train + export + đánh giá + so sánh 6 model mặc định
python -m pipelines.run_experiments --yolo-python .venv-yolo\Scripts\python.exe `
    --tf-python .venv-tf\Scripts\python.exe

# Chọn model, bỏ bước train
python -m pipelines.run_experiments --models yolo11n-cls mobilenetv3large `
    --steps export evaluate compare --splits val test test_ext

# Chạy thử đường ống trên tập rất nhỏ (1 epoch) — chỉ để kiểm tra lỗi
python -m pipelines.run_experiments --models yolo11n-cls mobilenetv3small --smoke
```

Mặc định: `yolo11n-cls`, `yolov8n-cls`, `mobilenetv2`, `mobilenetv3large`,
`efficientnetb0`, `efficientnet_lite0`. Mỗi bước chạy trong tiến trình riêng.

## 9. Đưa model vào app

```powershell
python -m pipelines.export_tflite --run yolo11n-cls_v2
python -m pipelines.evaluate --run yolo11n-cls_v2 --artifact tflite_float32 --split test test_ext
python -m tools.sync_flutter_assets --run yolo11n-cls_v2 --format float32
python -m tools.validate_chemsort_assets --project-root .. --require-flashcards `
    --export-report reports/experiments/yolo11n-cls_v2/export_report.json
cd ..; flutter test
```

`sync_flutter_assets` chỉ chép format đã PASS parity, cập nhật
`AppConfig.modelAsset` trong `lib/config.dart`, và từ chối model nhận pixel
0..255 (app luôn đưa 0..1; model Keras cần sửa `preprocessImage` trước).

## 10. Quy tắc

- `src/chemsort/contracts.py` là nguồn chuẩn duy nhất cho thứ tự lớp và mapping 9→6.
- `train` để học; `val` để chọn epoch / model / siêu tham số; `test` và
  `test_ext` chỉ dùng **sau khi** đã chọn model, để báo cáo. Không tune theo test.
- Không sửa tay manifest, model binary hay báo cáo; chạy lại script tương ứng.
- Đổi manifest ⇒ train, export và đánh giá lại toàn bộ.
- Model TFLite, `labels_9.txt` và `chemsort_mapping.json` luôn đi cùng nhau.
