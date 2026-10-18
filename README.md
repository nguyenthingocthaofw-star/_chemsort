# ChemSort — Scan, Classify, Discover, Learn

ChemSort là ứng dụng Flutter cho Android và iOS, giúp học sinh **phân loại rác tại nguồn** và
**học Hóa học từ chính món rác trong tay**. Chụp một vật thể → model nhận diện
vật liệu ngay trên điện thoại → app cho biết nhóm phân loại theo quy định Việt
Nam → mở một flashcard Hóa học về vật liệu đó.

```text
Home → Scan → Result ─┬─ Plastic → Chọn mã nhựa 1–7 → Polymer → Flashcard
                      └─ vật liệu khác                       → Flashcard
                                                                 └─ "Quét vật khác" → Scan
```

- **Nhận diện trên máy:** model YOLO11n-cls (TFLite, 6,2 MB) chạy trực tiếp trên
  điện thoại; ảnh không được gửi đi đâu.
- **6 nhóm vật liệu:** Giấy/Bìa carton, Thực phẩm, Thủy tinh, Kim loại, Nhựa,
  Khác.
- **Nhóm phân loại Việt Nam** theo Luật Bảo vệ Môi trường 2020 (Điều 75).
- **178 flashcard Hóa học** đã kiểm chứng, mỗi thẻ có nguồn tham khảo.

**Mục lục:** [Dataset](#1-dataset) · [Quy trình xử lý dữ liệu](#2-quy-trình-xử-lý-dữ-liệu)
· [Huấn luyện model](#3-huấn-luyện-model) · [Đánh giá](#4-đánh-giá)
· [Suy luận trong app](#5-suy-luận-trong-app) · [Nội dung flashcard](#6-nội-dung-flashcard)
· [Cấu trúc mã nguồn](#7-cấu-trúc-mã-nguồn) · [Chạy và build](#8-chạy-và-build)
· [Hạn chế](#9-hạn-chế) · [Nguồn dữ liệu và giấy phép](#10-nguồn-dữ-liệu-và-giấy-phép)

---

## 1. Dataset

### 1.1 Nguồn ảnh

| Nguồn | Nội dung | Giấy phép |
| --- | --- | --- |
| **RealWaste** (UCI, doi:10.24432/C5SS4G) | 4.752 ảnh rác thật chụp tại bãi rác, 9 lớp | CC BY-NC-SA 4.0 |
| **Drinking Waste Classification** (Kaggle, arkadiyhacks) | Lon nhôm, chai thủy tinh, chai nhựa HDPE/PET chụp riêng lẻ | CC0 1.0 |
| **TrashNet** (garythung) | Ảnh một vật thể trên nền trắng, 6 lớp | MIT |
| **Garbage Classification – 12 classes** (Kaggle, mostafaabla) | Ảnh web, 12 lớp | Bản quyền thuộc tác giả gốc; chỉ dùng cho mục đích học tập |

RealWaste là dataset gốc và **là nguồn duy nhất của tập val/test chính**. Ba
nguồn còn lại bổ sung ảnh **một vật thể riêng lẻ**, gần với cách người dùng chụp
bằng app, đặc biệt cho các lớp hay nhầm với nhau: Nhựa, Thủy tinh, Kim loại.

### 1.2 Hệ nhãn

Model học **9 lớp của RealWaste** (thứ tự cố định, khớp `labels_9.txt`). App gộp
9 lớp thành **6 nhóm vật liệu** (`chemsort_mapping.json`):

| Index | Lớp RealWaste | Nhóm trong app | Nhóm phân loại VN |
| --- | --- | --- | --- |
| 0 | Cardboard | Giấy / Bìa carton | Nhóm I – tái chế |
| 1 | Food Organics | Thực phẩm | Nhóm II – chất thải thực phẩm |
| 2 | Glass | Thủy tinh | Nhóm I – tái chế |
| 3 | Metal | Kim loại | Nhóm I – tái chế |
| 4 | Miscellaneous Trash | Khác | Chưa xác định |
| 5 | Paper | Giấy / Bìa carton | Nhóm I – tái chế |
| 6 | Plastic | Nhựa | Nhóm I – tái chế |
| 7 | Textile Trash | Khác | Chưa xác định |
| 8 | Vegetation | Khác | Chưa xác định |

Nhóm **Khác** cố ý **không** được gán Nhóm III: lớp này gom vải, lá cây và rác
hỗn hợp, nên app hiển thị "Chưa đủ thông tin" thay vì đoán sai nhóm.

Ánh xạ nhãn từ các nguồn bổ sung về 9 lớp RealWaste:

| Nguồn | Nhãn gốc → Lớp RealWaste |
| --- | --- |
| Drinking Waste | AluCan → Metal · Glass → Glass · HDPEM, PET → Plastic |
| TrashNet | cardboard, glass, metal, paper, plastic → cùng tên · trash → Miscellaneous Trash |
| Garbage Classification | biological → Food Organics · brown/green/white-glass → Glass · clothes, shoes → Textile Trash · battery, trash → Miscellaneous Trash · cardboard, metal, paper, plastic → cùng tên |

### 1.3 Phân chia dữ liệu (phiên bản 2.0.0)

| Lớp | Train | Val | Test | Test_ext | Tổng |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cardboard | 1.043 | 69 | 69 | 79 | 1.260 |
| Food Organics | 648 | 61 | 62 | 40 | 811 |
| Glass | 1.088 | 63 | 63 | 88 | 1.302 |
| Metal | 1.504 | 119 | 118 | 103 | 1.844 |
| Miscellaneous Trash | 829 | 74 | 75 | 54 | 1.032 |
| Paper | 1.070 | 75 | 75 | 80 | 1.300 |
| Plastic | 1.648 | 138 | 138 | 110 | 2.034 |
| Textile Trash | 582 | 48 | 48 | 40 | 718 |
| Vegetation | 305 | 66 | 65 | 0 | 436 |
| **Tổng** | **8.717** | **713** | **713** | **594** | **10.737** |

Theo nguồn:

| Nguồn | Train | Val | Test | Test_ext |
| --- | ---: | ---: | ---: | ---: |
| RealWaste | 3.326 | 713 | 713 | — |
| Drinking Waste | 1.080 | — | — | 120 |
| TrashNet | 1.482 | — | — | 163 |
| Garbage Classification | 2.829 | — | — | 311 |

- **Val / Test** chỉ gồm ảnh RealWaste (ảnh rác thật, nhiều vật thể, nền phức tạp).
- **Test_ext** là ảnh một vật thể từ ba nguồn bổ sung, giữ riêng để đo khả năng
  nhận diện ảnh chụp kiểu app. Không dùng để train hay chọn model.
- Vegetation không có trong các nguồn bổ sung nên không có ảnh test_ext.

---

## 2. Quy trình xử lý dữ liệu

### 2.1 Phiên bản 1 — làm sạch và chia RealWaste

1. **Kiểm kê:** quét toàn bộ ảnh trong 9 thư mục lớp, đối chiếu số lượng với số
   công bố của RealWaste.
2. **Kiểm tra ảnh lỗi:** mở từng ảnh bằng Pillow; ảnh không đọc được ghi vào danh
   sách lỗi (kết quả: 0 ảnh lỗi).
3. **Lấy dấu vân tay:** với mỗi ảnh lưu kích thước, mode màu, dung lượng,
   **SHA-256** (trùng tuyệt đối) và **perceptual hash – pHash** (gần giống).
4. **Loại trùng tuyệt đối** theo SHA-256.
5. **Loại thủ công** các ảnh nhóm đánh giá là không phù hợp (danh sách loại trừ
   có ghi lý do).
6. **Chia stratified** train/val/test theo tỷ lệ **70/15/15**, seed **42**, giữ tỷ
   lệ lớp như nhau ở cả ba tập.
7. **Kiểm tra rò rỉ giữa các tập:** liệt kê các cặp ảnh có khoảng cách Hamming
   pHash ≤ 4 và đánh dấu cặp nằm ở hai tập khác nhau để rà soát.
8. **Khóa phiên bản:** ghi manifest (đường dẫn, nhãn, tập, kích thước, SHA-256,
   pHash), thống kê theo tập và checksum của manifest. Split này **giữ cố định**
   cho mọi lần so sánh model về sau.

### 2.2 Phiên bản 2 — bổ sung ảnh một vật thể

Model chỉ học RealWaste nhận diện tốt ảnh bãi rác nhưng kém với ảnh chụp **một
vật thể riêng lẻ**, đúng kiểu người dùng chụp bằng app. Phiên bản 2 bổ sung ba
nguồn ảnh theo các quy tắc:

1. **Giữ nguyên split RealWaste của v1:** val và test của v1 và v2 giống hệt
   nhau, nên so sánh giữa các phiên bản là công bằng.
2. **Ánh xạ nhãn** của từng nguồn về 9 lớp RealWaste (bảng ở mục 1.2).
3. **Loại trùng tuyệt đối** (SHA-256) với RealWaste và giữa các nguồn với nhau:
   hai bộ Kaggle chứa lại nhiều ảnh TrashNet.
4. **Loại ảnh gần giống bất kỳ ảnh RealWaste nào** (pHash ≤ 4), để không ảnh bổ
   sung nào "nhìn lén" được val/test.
5. **Giới hạn 400 ảnh cho mỗi cặp (nguồn, lớp)** để một nguồn không lấn át cả
   một lớp.
6. **Tách 10% làm test_ext**, stratified theo (nguồn, lớp); phần còn lại vào train.
7. **Loại ảnh test_ext có ảnh gần giống trong train** (pHash ≤ 4).

Số ảnh qua từng bước lọc:

| Bước | Số ảnh |
| --- | ---: |
| Đọc được từ ba nguồn bổ sung | 22.862 |
| Sau loại trùng tuyệt đối | 19.731 |
| Sau loại ảnh gần giống RealWaste | 19.721 |
| Sau giới hạn 400 ảnh/(nguồn, lớp) | 5.991 |
| Bị loại khỏi test_ext vì gần giống ảnh train | 6 |
| **Kết quả:** train bổ sung / test_ext | **5.391 / 594** |

---

## 3. Huấn luyện model

| Thiết lập | Giá trị |
| --- | --- |
| Kiến trúc | **YOLO11n-cls** (Ultralytics), 1,54 triệu tham số, 3,2 GFLOPs |
| Khởi tạo | Trọng số pretrained ImageNet, **fine-tune toàn bộ mạng** |
| Dữ liệu | Train v2 (8.717 ảnh), chọn epoch tốt nhất theo top-1 trên val |
| Kích thước ảnh | 224 × 224 |
| Epoch / batch | 60 / 64, early stopping patience 15 |
| Augmentation | RandAugment, random erasing 0,4, lật ngang 0,5, dao động màu HSV |
| Seed | 42, chế độ deterministic |
| Phần cứng | NVIDIA RTX 3060 Laptop (6 GB), khoảng 1 giờ |

Ultralytics đọc dữ liệu dạng thư mục `split/lớp/ảnh`. Thư mục này được dựng từ
manifest bằng hard link, nên không tốn thêm dung lượng đĩa. Ultralytics xếp lớp
theo thứ tự chữ cái, trùng với thứ tự 9 lớp ở mục 1.2; bước export kiểm tra lại
thứ tự này trước khi xuất.

**Xuất model:** checkpoint tốt nhất được export sang **TFLite float32** qua
LiteRT (6,2 MB). Đầu vào `[1, 3, 224, 224]` (NCHW, RGB, giá trị 0–1), đầu ra
`[1, 9]` là xác suất softmax.

---

## 4. Đánh giá

Điểm được đo trên **chính file TFLite đóng gói trong app**, với **đúng các bước
tiền xử lý của app** (mục 5). Vì vậy đây là con số điện thoại thực sự đạt được,
không phải điểm của checkpoint PyTorch.

| Tập | Số ảnh | Accuracy 6 nhóm | **Macro-F1 6 nhóm** | Macro-F1 9 lớp |
| --- | ---: | ---: | ---: | ---: |
| Test (RealWaste) | 713 | 0,922 | **0,917** | 0,913 |
| Test_ext (ảnh một vật thể) | 594 | 0,933 | **0,926** | 0,818\* |

\* Test_ext không có lớp Vegetation, nên macro-F1 9 lớp bị kéo xuống bởi một lớp
không có mẫu. Chỉ số 6 nhóm mới là chỉ số chính.

F1 từng nhóm:

| Nhóm | Test (RealWaste) | Test_ext |
| --- | ---: | ---: |
| Giấy / Bìa carton | 0,956 | 0,972 |
| Thực phẩm | 0,898 | 0,914 |
| Thủy tinh | 0,917 | 0,936 |
| Kim loại | 0,912 | 0,938 |
| Khác | 0,939 | 0,868 |
| Nhựa | 0,881 | 0,929 |

**Đã thử trước khi chọn YOLO11n-cls:** EfficientNet-Lite0 chỉ train phần đầu
phân loại (đóng băng backbone), trên cùng dataset v2, đạt macro-F1 6 nhóm
**0,850** trên test và **0,911** trên test_ext, file 12,9 MB. YOLO11n-cls tốt hơn
trên cả hai tập và nhẹ bằng một nửa.

Tập val đã dùng để chọn epoch nên không được dùng làm điểm cuối cùng.

---

## 5. Suy luận trong app

1. **Chụp/chọn ảnh:** ảnh được thu nhỏ còn tối đa 1024 px ngay khi lấy.
2. **Tiền xử lý** (`preprocessImage` trong `lib/services/yolo_classifier.dart`):
   xoay theo EXIF → RGB → **cắt vuông ở giữa** → **thu nhỏ trung bình vùng** về
   224 × 224 → chia 255 → đổi sang thứ tự NCHW.
3. **Chạy model TFLite** (4 luồng CPU), nhận 9 xác suất.
4. **Gộp 9 lớp → 6 nhóm:** cộng xác suất các lớp trong cùng nhóm rồi lấy nhóm
   cao nhất (`mapping_service.dart`). Ví dụ Paper 0,20 + Cardboard 0,25 = 0,45
   sẽ thắng Plastic 0,30.
5. **Ngưỡng tin cậy 0,5:** nhóm thắng dưới 50% thì app yêu cầu chụp lại ("Hãy
   chụp gần hơn, đủ sáng và chỉ một vật thể") thay vì đưa ra kết quả không chắc.

Cắt vuông ở giữa rồi thu nhỏ trung bình vùng cho cùng khung hình với cách
Ultralytics đánh giá (resize cạnh ngắn rồi cắt giữa), đồng thời tránh răng cưa
khi thu nhỏ ảnh lớn.

---

## 6. Nội dung flashcard

Flashcard **không do AI sinh ra**. Model chỉ chọn vật liệu; nội dung lấy từ thư
viện 178 fact đã được nhóm viết và rà soát (`review_status = reviewed`), mỗi
fact có tiêu đề nguồn và đường link.

| Nhóm | Số fact |
| --- | ---: |
| Nhựa (8 fact cho mỗi mã nhựa 1–7, cộng fact chung) | 76 |
| Thực phẩm | 21 |
| Khác | 21 |
| Giấy / Bìa carton | 20 |
| Thủy tinh | 20 |
| Kim loại | 20 |

Nguồn chính: OpenStax (Chemistry 2e, Organic Chemistry, Biology 2e, Concepts of
Biology), US EPA, EdTech Books, USDA Forest Products Laboratory, Corning Museum
of Glass, NIST, Library of Congress, Cornell Composting, US Forest Service.

Quy tắc hiển thị:

- Mỗi lần quét chỉ xem **một** fact. Không có nút "Fact khác"; nút Back từ
  flashcard quay thẳng về màn Scan.
- Fact **không lặp liên tiếp**: mỗi nhóm (vật liệu hoặc mã nhựa) có một bộ bài
  xáo trộn, phải xem hết bộ mới quay lại fact cũ.
- Với Nhựa, người dùng chọn **mã nhựa 1–7** (PET, HDPE, PVC, LDPE, PP, PS,
  Other) in trên sản phẩm để xem fact về đúng polymer đó.

---

## 7. Cấu trúc mã nguồn

```text
lib/
  main.dart                   Khởi động app, nạp model + flashcard
  config.dart                 Đường dẫn asset, ngưỡng tin cậy
  theme.dart                  Màu sắc, kiểu nút
  models/
    waste_material.dart       6 nhóm vật liệu + nhóm phân loại Việt Nam
    resin.dart                7 mã nhựa
    flashcard.dart            Một fact
    prediction.dart           Kết quả nhận diện
  services/
    yolo_classifier.dart      Chạy TFLite + tiền xử lý ảnh
    mapping_service.dart      Gộp 9 lớp thành 6 nhóm
    fact_service.dart         Chọn fact, không lặp liên tiếp
    app_services.dart         Nạp các service một lần khi mở app
  screens/                    Home, Scan, Result, Resin, Polymer, Flashcard
  widgets/                    Huy hiệu nhóm, biểu tượng mã nhựa
assets/chemsort/
  yolo11n-cls_data_v2_float32.tflite   Model (6,2 MB)
  labels_9.txt                         Thứ tự 9 lớp
  chemsort_mapping.json                Gộp 9 lớp → 6 nhóm
  flashcards.json                      178 fact
test/                         Unit test mapping, tiền xử lý ảnh, flashcard, widget
```

---

## 8. Chạy và build

Yêu cầu chung: Flutter SDK 3.44 trở lên (Dart ≥ 3.12).

### Android

Cần Android Studio + Android SDK và điện thoại Android 8.0 trở lên (API 26+)
đã bật USB debugging.

```powershell
flutter pub get
flutter test
flutter run
```

Build APK và cài vào điện thoại:

```powershell
flutter build apk --release --split-per-abi
adb install -r build\app\outputs\flutter-apk\app-arm64-v8a-release.apk
```

- `adb` nằm trong `%LOCALAPPDATA%\Android\Sdk\platform-tools\`; nếu chưa có trong
  PATH thì gọi bằng đường dẫn đầy đủ.
- Windows 11 bật **Smart App Control** có thể chặn bước build release; khi đó
  hãy build trong WSL hoặc Linux.
- Điện thoại Xiaomi/Redmi/POCO cần bật **"Cài đặt qua USB"** trong Tùy chọn nhà
  phát triển thì `adb install` mới chạy được.

### iOS

Cần macOS, Xcode 15 trở lên, CocoaPods và iPhone chạy iOS 15 trở lên. Project
dùng CocoaPods vì `tflite_flutter 0.12.1` chưa cung cấp Swift Package Manager
manifest cho phần native iOS.

```bash
flutter pub get
cd ios && pod install && cd ..
flutter test
flutter run
# Hoặc build bản không ký để kiểm tra quá trình biên dịch:
flutter build ios --release --no-codesign
```

Các mô tả quyền camera và thư viện ảnh đã có trong `ios/Runner/Info.plist`.
`tflite_flutter` có thể không chạy đúng trên iOS Simulator; nên kiểm tra luồng
chụp/chọn ảnh và suy luận trên iPhone thật trước khi phát hành. Build archive
(`flutter build ipa`) cũng cần cấu hình signing/team trong Xcode. Project đã đặt
`Strip Style` thành `Non-Global Symbols` cho Release/Profile để tránh loại bỏ các
symbol C mà TFLite tra cứu khi chạy.

Nếu vừa chuyển từ cấu hình Swift Package Manager cũ, hãy làm sạch artifact trước
khi build lại:

```bash
flutter clean
rm -rf ios/Pods ios/Podfile.lock
flutter pub get
cd ios && pod install --repo-update && cd ..
flutter build ios --release --no-codesign
```

---

## 9. Hạn chế

- Test và test_ext đều là ảnh từ các dataset công khai. Test_ext lấy từ cùng
  nguồn với ảnh bổ sung trong train, nên điểm 0,926 có thể hơi lạc quan. Cần
  kiểm chứng thêm bằng ảnh nhóm tự chụp ở điều kiện thực tế.
- Model nhận diện **vật thể chính** trong ảnh; ảnh nhiều vật thể hoặc vật làm từ
  nhiều vật liệu dễ bị xếp vào nhóm Khác hoặc bị yêu cầu chụp lại.
- Nhựa và Thực phẩm là hai nhóm có F1 thấp nhất trên test RealWaste (0,88 và 0,90).
- App không tự đọc mã nhựa trên sản phẩm; người dùng tự chọn mã 1–7.

---

## 10. Nguồn dữ liệu và giấy phép

- **RealWaste:** S. Single, S. Iranmanesh, R. Raad. *RealWaste: A Novel
  Real-Life Data Set for Landfill Waste Classification Using Deep Learning.*
  Information, 2023. UCI Machine Learning Repository, doi:10.24432/C5SS4G.
  CC BY-NC-SA 4.0.
- **Drinking Waste Classification:** Kaggle (arkadiyhacks), CC0 1.0.
- **TrashNet:** G. Thung, M. Yang. github.com/garythung/trashnet, MIT.
- **Garbage Classification (12 classes):** Kaggle (mostafaabla). Ảnh thu thập
  từ web, bản quyền thuộc tác giả gốc; chỉ dùng cho mục đích học tập.
- **YOLO11:** Ultralytics, AGPL-3.0.

Do RealWaste dùng giấy phép **phi thương mại** (CC BY-NC-SA 4.0) và Ultralytics
dùng AGPL-3.0, model và app này chỉ dành cho **mục đích giáo dục, phi thương mại**.
