# Model comparison - split `test_ext`

Sorted by `macro_f1_6class`. 6 = app groups (probabilities summed per group), 9 = RealWaste classes.
Macro averages weight every class equally; CI = 95% bootstrap interval (2000 resamples).

| # | Model | Images | Acc 6 | Prec 6 | Rec 6 | F1 6 | wF1 6 | Kappa 6 | MCC 6 | Acc 9 | Prec 9 | Rec 9 | F1 9 | Top3 9 | ECE 6 | Size MB | ms/img |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | app_yolo11n-cls_v2 [tflite] | 594 | 0.9327 | 0.9270 | 0.9260 | 0.9260 | 0.9323 | 0.9171 | 0.9172 | 0.9276 | 0.8210 | 0.8166 | 0.8183 | 0.9882 | 0.0413 | 5.94 | 13.3 |
| 2 | yolo11n-cls_v2_aug_medium [pytorch] | 594 | 0.9242 | 0.9218 | 0.9171 | 0.9187 | 0.9237 | 0.9066 | 0.9070 | 0.9226 | 0.8184 | 0.8115 | 0.8137 | 0.9832 | 0.0510 | 3.06 | 3.0 |
| 3 | mobilenetv2_v2_aug_medium [keras] | 594 | 0.9040 | 0.9103 | 0.9020 | 0.9056 | 0.9035 | 0.8816 | 0.8819 | 0.8906 | 0.8004 | 0.7917 | 0.7944 | 0.9832 | 0.0394 | 20.69 | 13.9 |
| 4 | mobilenetv2_v2_aug_medium [tflite_float16] | 594 | 0.9024 | 0.9076 | 0.9005 | 0.9037 | 0.9018 | 0.8796 | 0.8798 | 0.8889 | 0.7961 | 0.7907 | 0.7924 | 0.9832 | 0.0380 | 4.33 | 3.4 |

## Macro-F1 (6 groups) with 95% CI

- **app_yolo11n-cls_v2 [tflite]**: 0.9260 [0.9014, 0.9472]
- **yolo11n-cls_v2_aug_medium [pytorch]**: 0.9187 [0.8941, 0.9419]
- **mobilenetv2_v2_aug_medium [keras]**: 0.9056 [0.8809, 0.9281]
- **mobilenetv2_v2_aug_medium [tflite_float16]**: 0.9037 [0.8786, 0.9268]

## Per-class F1 - 6 app groups

| Model | Paper/Cardboard | Organic | Glass | Metal | Other | Plastic |
|---|---:|---:|---:|---:|---:|---:|
| app_yolo11n-cls_v2 [tflite] | 0.972 | 0.914 | 0.936 | 0.938 | 0.868 | 0.929 |
| yolo11n-cls_v2_aug_medium [pytorch] | 0.969 | 0.925 | 0.924 | 0.937 | 0.844 | 0.913 |
| mobilenetv2_v2_aug_medium [keras] | 0.930 | 0.962 | 0.875 | 0.918 | 0.849 | 0.900 |
| mobilenetv2_v2_aug_medium [tflite_float16] | 0.933 | 0.962 | 0.880 | 0.913 | 0.835 | 0.899 |

## Per-class F1 - 9 RealWaste classes

| Model | Cardboard | Food Organics | Glass | Metal | Miscellaneous Trash | Paper | Plastic | Textile Trash | Vegetation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| app_yolo11n-cls_v2 [tflite] | 0.974 | 0.914 | 0.936 | 0.938 | 0.804 | 0.945 | 0.929 | 0.925 | 0.000 |
| yolo11n-cls_v2_aug_medium [pytorch] | 0.987 | 0.925 | 0.924 | 0.937 | 0.796 | 0.938 | 0.913 | 0.902 | 0.000 |
| mobilenetv2_v2_aug_medium [keras] | 0.904 | 0.962 | 0.875 | 0.913 | 0.750 | 0.869 | 0.900 | 0.976 | 0.000 |
| mobilenetv2_v2_aug_medium [tflite_float16] | 0.904 | 0.962 | 0.880 | 0.909 | 0.727 | 0.874 | 0.899 | 0.976 | 0.000 |

## McNemar tests (6 groups)

p < 0.05 means the two models' error rates differ significantly on the same images.

| Model A | Model B | Only A correct | Only B correct | p-value |
|---|---|---:|---:|---:|
| app_yolo11n-cls_v2 [tflite] | yolo11n-cls_v2_aug_medium [pytorch] | 17 | 12 | 0.4576 |
| app_yolo11n-cls_v2 [tflite] | mobilenetv2_v2_aug_medium [keras] | 33 | 16 | 0.02227 |
| app_yolo11n-cls_v2 [tflite] | mobilenetv2_v2_aug_medium [tflite_float16] | 34 | 16 | 0.01621 |
| yolo11n-cls_v2_aug_medium [pytorch] | mobilenetv2_v2_aug_medium [keras] | 33 | 21 | 0.1344 |
| yolo11n-cls_v2_aug_medium [pytorch] | mobilenetv2_v2_aug_medium [tflite_float16] | 34 | 21 | 0.1056 |
| mobilenetv2_v2_aug_medium [keras] | mobilenetv2_v2_aug_medium [tflite_float16] | 1 | 0 | 1 |
