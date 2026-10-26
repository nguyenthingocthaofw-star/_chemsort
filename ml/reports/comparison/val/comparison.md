# Model comparison - split `val`

Sorted by `macro_f1_6class`. 6 = app groups (probabilities summed per group), 9 = RealWaste classes.
Macro averages weight every class equally; CI = 95% bootstrap interval (2000 resamples).

| # | Model | Images | Acc 6 | Prec 6 | Rec 6 | F1 6 | wF1 6 | Kappa 6 | MCC 6 | Acc 9 | Prec 9 | Rec 9 | F1 9 | Top3 9 | ECE 6 | Size MB | ms/img |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | app_yolo11n-cls_v2 [tflite] | 713 | 0.9481 | 0.9510 | 0.9458 | 0.9482 | 0.9481 | 0.9359 | 0.9359 | 0.9341 | 0.9349 | 0.9317 | 0.9328 | 0.9916 | 0.0303 | 5.94 | 14.7 |
| 2 | yolo11n-cls_v2_aug_medium [pytorch] | 713 | 0.9397 | 0.9432 | 0.9327 | 0.9374 | 0.9398 | 0.9254 | 0.9256 | 0.9243 | 0.9297 | 0.9199 | 0.9244 | 0.9930 | 0.0264 | 3.06 | 2.5 |
| 3 | mobilenetv2_v2_aug_medium [keras] | 713 | 0.8177 | 0.8429 | 0.7953 | 0.8119 | 0.8198 | 0.7736 | 0.7763 | 0.7756 | 0.8038 | 0.7648 | 0.7721 | 0.9607 | 0.0589 | 20.69 | 14.7 |
| 4 | mobilenetv2_v2_aug_medium [tflite_float16] | 713 | 0.8177 | 0.8451 | 0.7941 | 0.8117 | 0.8198 | 0.7735 | 0.7764 | 0.7770 | 0.8063 | 0.7655 | 0.7737 | 0.9593 | 0.0593 | 4.33 | 3.5 |

## Macro-F1 (6 groups) with 95% CI

- **app_yolo11n-cls_v2 [tflite]**: 0.9482 [0.9306, 0.9647]
- **yolo11n-cls_v2_aug_medium [pytorch]**: 0.9374 [0.9164, 0.9552]
- **mobilenetv2_v2_aug_medium [keras]**: 0.8119 [0.7782, 0.8424]
- **mobilenetv2_v2_aug_medium [tflite_float16]**: 0.8117 [0.7783, 0.8427]

## Per-class F1 - 6 app groups

| Model | Paper/Cardboard | Organic | Glass | Metal | Other | Plastic |
|---|---:|---:|---:|---:|---:|---:|
| app_yolo11n-cls_v2 [tflite] | 0.959 | 0.958 | 0.944 | 0.941 | 0.952 | 0.936 |
| yolo11n-cls_v2_aug_medium [pytorch] | 0.976 | 0.957 | 0.903 | 0.928 | 0.944 | 0.916 |
| mobilenetv2_v2_aug_medium [keras] | 0.863 | 0.778 | 0.800 | 0.825 | 0.853 | 0.752 |
| mobilenetv2_v2_aug_medium [tflite_float16] | 0.863 | 0.778 | 0.796 | 0.830 | 0.853 | 0.750 |

## Per-class F1 - 9 RealWaste classes

| Model | Cardboard | Food Organics | Glass | Metal | Miscellaneous Trash | Paper | Plastic | Textile Trash | Vegetation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| app_yolo11n-cls_v2 [tflite] | 0.944 | 0.958 | 0.944 | 0.941 | 0.881 | 0.925 | 0.936 | 0.896 | 0.970 |
| yolo11n-cls_v2_aug_medium [pytorch] | 0.971 | 0.957 | 0.903 | 0.928 | 0.871 | 0.927 | 0.916 | 0.884 | 0.962 |
| mobilenetv2_v2_aug_medium [keras] | 0.648 | 0.778 | 0.800 | 0.818 | 0.632 | 0.795 | 0.756 | 0.819 | 0.903 |
| mobilenetv2_v2_aug_medium [tflite_float16] | 0.661 | 0.778 | 0.796 | 0.826 | 0.632 | 0.795 | 0.754 | 0.819 | 0.903 |

## McNemar tests (6 groups)

p < 0.05 means the two models' error rates differ significantly on the same images.

| Model A | Model B | Only A correct | Only B correct | p-value |
|---|---|---:|---:|---:|
| app_yolo11n-cls_v2 [tflite] | yolo11n-cls_v2_aug_medium [pytorch] | 19 | 13 | 0.3768 |
| app_yolo11n-cls_v2 [tflite] | mobilenetv2_v2_aug_medium [keras] | 106 | 13 | 3.351e-17 |
| app_yolo11n-cls_v2 [tflite] | mobilenetv2_v2_aug_medium [tflite_float16] | 106 | 13 | 3.351e-17 |
| yolo11n-cls_v2_aug_medium [pytorch] | mobilenetv2_v2_aug_medium [keras] | 103 | 16 | 3.181e-15 |
| yolo11n-cls_v2_aug_medium [pytorch] | mobilenetv2_v2_aug_medium [tflite_float16] | 104 | 17 | 5.359e-15 |
| mobilenetv2_v2_aug_medium [keras] | mobilenetv2_v2_aug_medium [tflite_float16] | 1 | 1 | 1 |
