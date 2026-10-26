# Model comparison - split `test`

Sorted by `macro_f1_6class`. 6 = app groups (probabilities summed per group), 9 = RealWaste classes.
Macro averages weight every class equally; CI = 95% bootstrap interval (2000 resamples).

> Warning: not every model was scored on the same images (e.g. a report made from an
> older manifest). Headline metrics are not strictly comparable; McNemar tests use
> only the shared images.
>
> - legacy_v1_efficientnet_lite0 [tflite_float16]: 683/713 images shared with the largest set
> - legacy_v1_mobilenetv2_finetuned [keras]: 683/713 images shared with the largest set


| # | Model | Images | Acc 6 | Prec 6 | Rec 6 | F1 6 | wF1 6 | Kappa 6 | MCC 6 | Acc 9 | Prec 9 | Rec 9 | F1 9 | Top3 9 | ECE 6 | Size MB | ms/img |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | yolo11n-cls_v2_aug_medium [pytorch] | 713 | 0.9285 | 0.9258 | 0.9264 | 0.9258 | 0.9282 | 0.9117 | 0.9117 | 0.9102 | 0.9108 | 0.9142 | 0.9121 | 0.9874 | 0.0413 | 3.06 | 3.7 |
| 2 | app_yolo11n-cls_v2 [tflite] | 713 | 0.9215 | 0.9179 | 0.9184 | 0.9171 | 0.9211 | 0.9030 | 0.9032 | 0.9088 | 0.9106 | 0.9177 | 0.9127 | 0.9902 | 0.0526 | 5.94 | 13.9 |
| 3 | mobilenetv3large_v2_aug_medium [tflite_float16] | 713 | 0.8710 | 0.8646 | 0.8729 | 0.8679 | 0.8705 | 0.8408 | 0.8411 | 0.8443 | 0.8472 | 0.8538 | 0.8495 | 0.9762 | 0.0357 | 5.79 | 3.4 |
| 4 | mobilenetv3large_v2_aug_medium [keras] | 713 | 0.8696 | 0.8622 | 0.8715 | 0.8660 | 0.8692 | 0.8391 | 0.8394 | 0.8429 | 0.8456 | 0.8529 | 0.8482 | 0.9776 | 0.0293 | 24.12 | 14.1 |
| 5 | legacy_v1_efficientnet_lite0 [tflite_float16] | 713 | 0.8583 | 0.8553 | 0.8476 | 0.8500 | 0.8578 | 0.8247 | 0.8250 | 0.8247 | 0.8275 | 0.8344 | 0.8298 | 0.9748 | 0.0270 | 6.47 |  |
| 6 | efficientnetb0_v2_aug_medium [keras] | 713 | 0.8527 | 0.8543 | 0.8438 | 0.8484 | 0.8527 | 0.8178 | 0.8180 | 0.8289 | 0.8408 | 0.8430 | 0.8359 | 0.9790 | 0.0367 | 27.50 | 24.9 |
| 7 | efficientnetb0_v2_aug_medium [tflite_float16] | 713 | 0.8527 | 0.8542 | 0.8438 | 0.8484 | 0.8526 | 0.8178 | 0.8180 | 0.8289 | 0.8408 | 0.8430 | 0.8359 | 0.9790 | 0.0370 | 7.78 | 8.2 |
| 8 | legacy_v1_mobilenetv2_finetuned [keras] | 713 | 0.8401 | 0.8352 | 0.8282 | 0.8304 | 0.8398 | 0.8020 | 0.8027 | 0.8205 | 0.8286 | 0.8358 | 0.8310 | 0.9762 | 0.0495 |  |  |
| 9 | mobilenetv2_v2_aug_medium [keras] | 713 | 0.8219 | 0.8380 | 0.8044 | 0.8165 | 0.8238 | 0.7791 | 0.7811 | 0.7966 | 0.8211 | 0.7955 | 0.8002 | 0.9649 | 0.0474 | 20.69 | 14.1 |
| 10 | mobilenetv2_v2_aug_medium [tflite_float16] | 713 | 0.8191 | 0.8374 | 0.7989 | 0.8128 | 0.8211 | 0.7755 | 0.7778 | 0.7938 | 0.8202 | 0.7918 | 0.7975 | 0.9649 | 0.0535 | 4.33 | 3.6 |

## Macro-F1 (6 groups) with 95% CI

- **yolo11n-cls_v2_aug_medium [pytorch]**: 0.9258 [0.9049, 0.9452]
- **app_yolo11n-cls_v2 [tflite]**: 0.9171 [0.8940, 0.9368]
- **mobilenetv3large_v2_aug_medium [tflite_float16]**: 0.8679 [0.8420, 0.8935]
- **mobilenetv3large_v2_aug_medium [keras]**: 0.8660 [0.8401, 0.8919]
- **legacy_v1_efficientnet_lite0 [tflite_float16]**: 0.8500 [0.8211, 0.8773]
- **efficientnetb0_v2_aug_medium [keras]**: 0.8484 [0.8184, 0.8757]
- **efficientnetb0_v2_aug_medium [tflite_float16]**: 0.8484 [0.8184, 0.8757]
- **legacy_v1_mobilenetv2_finetuned [keras]**: 0.8304 [0.7995, 0.8584]
- **mobilenetv2_v2_aug_medium [keras]**: 0.8165 [0.7838, 0.8450]
- **mobilenetv2_v2_aug_medium [tflite_float16]**: 0.8128 [0.7806, 0.8416]

## Per-class F1 - 6 app groups

| Model | Paper/Cardboard | Organic | Glass | Metal | Other | Plastic |
|---|---:|---:|---:|---:|---:|---:|
| yolo11n-cls_v2_aug_medium [pytorch] | 0.966 | 0.909 | 0.930 | 0.924 | 0.936 | 0.890 |
| app_yolo11n-cls_v2 [tflite] | 0.956 | 0.898 | 0.917 | 0.912 | 0.939 | 0.881 |
| mobilenetv3large_v2_aug_medium [tflite_float16] | 0.919 | 0.916 | 0.816 | 0.852 | 0.889 | 0.815 |
| mobilenetv3large_v2_aug_medium [keras] | 0.919 | 0.916 | 0.810 | 0.848 | 0.889 | 0.815 |
| legacy_v1_efficientnet_lite0 [tflite_float16] | 0.911 | 0.855 | 0.812 | 0.830 | 0.896 | 0.796 |
| efficientnetb0_v2_aug_medium [keras] | 0.919 | 0.874 | 0.823 | 0.809 | 0.885 | 0.782 |
| efficientnetb0_v2_aug_medium [tflite_float16] | 0.916 | 0.874 | 0.823 | 0.809 | 0.885 | 0.784 |
| legacy_v1_mobilenetv2_finetuned [keras] | 0.937 | 0.840 | 0.758 | 0.829 | 0.856 | 0.763 |
| mobilenetv2_v2_aug_medium [keras] | 0.879 | 0.804 | 0.777 | 0.834 | 0.841 | 0.765 |
| mobilenetv2_v2_aug_medium [tflite_float16] | 0.879 | 0.804 | 0.763 | 0.829 | 0.841 | 0.762 |

## Per-class F1 - 9 RealWaste classes

| Model | Cardboard | Food Organics | Glass | Metal | Miscellaneous Trash | Paper | Plastic | Textile Trash | Vegetation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11n-cls_v2_aug_medium [pytorch] | 0.923 | 0.909 | 0.930 | 0.924 | 0.849 | 0.926 | 0.890 | 0.918 | 0.938 |
| app_yolo11n-cls_v2 [tflite] | 0.917 | 0.898 | 0.917 | 0.912 | 0.861 | 0.926 | 0.881 | 0.941 | 0.961 |
| mobilenetv3large_v2_aug_medium [tflite_float16] | 0.844 | 0.916 | 0.816 | 0.849 | 0.708 | 0.878 | 0.809 | 0.863 | 0.961 |
| mobilenetv3large_v2_aug_medium [keras] | 0.844 | 0.916 | 0.810 | 0.844 | 0.708 | 0.878 | 0.809 | 0.863 | 0.961 |
| legacy_v1_efficientnet_lite0 [tflite_float16] | 0.803 | 0.855 | 0.812 | 0.841 | 0.728 | 0.838 | 0.793 | 0.851 | 0.947 |
| efficientnetb0_v2_aug_medium [keras] | 0.853 | 0.867 | 0.823 | 0.809 | 0.736 | 0.849 | 0.782 | 0.862 | 0.942 |
| efficientnetb0_v2_aug_medium [tflite_float16] | 0.853 | 0.867 | 0.823 | 0.809 | 0.736 | 0.849 | 0.782 | 0.862 | 0.942 |
| legacy_v1_mobilenetv2_finetuned [keras] | 0.935 | 0.850 | 0.758 | 0.826 | 0.667 | 0.881 | 0.758 | 0.880 | 0.925 |
| mobilenetv2_v2_aug_medium [keras] | 0.824 | 0.814 | 0.777 | 0.831 | 0.611 | 0.819 | 0.758 | 0.862 | 0.906 |
| mobilenetv2_v2_aug_medium [tflite_float16] | 0.824 | 0.814 | 0.763 | 0.829 | 0.611 | 0.814 | 0.755 | 0.862 | 0.906 |

## McNemar tests (6 groups)

p < 0.05 means the two models' error rates differ significantly on the same images.

| Model A | Model B | Only A correct | Only B correct | p-value |
|---|---|---:|---:|---:|
| yolo11n-cls_v2_aug_medium [pytorch] | app_yolo11n-cls_v2 [tflite] | 21 | 16 | 0.5108 |
| yolo11n-cls_v2_aug_medium [pytorch] | mobilenetv3large_v2_aug_medium [tflite_float16] | 63 | 22 | 1.434e-05 |
| yolo11n-cls_v2_aug_medium [pytorch] | mobilenetv3large_v2_aug_medium [keras] | 64 | 22 | 9.818e-06 |
| yolo11n-cls_v2_aug_medium [pytorch] | legacy_v1_efficientnet_lite0 [tflite_float16] | 68 | 23 | 3.979e-06 |
| yolo11n-cls_v2_aug_medium [pytorch] | efficientnetb0_v2_aug_medium [keras] | 73 | 19 | 3.283e-08 |
| yolo11n-cls_v2_aug_medium [pytorch] | efficientnetb0_v2_aug_medium [tflite_float16] | 73 | 19 | 3.283e-08 |
| yolo11n-cls_v2_aug_medium [pytorch] | legacy_v1_mobilenetv2_finetuned [keras] | 77 | 18 | 2.67e-09 |
| yolo11n-cls_v2_aug_medium [pytorch] | mobilenetv2_v2_aug_medium [keras] | 95 | 19 | 2.15e-12 |
| yolo11n-cls_v2_aug_medium [pytorch] | mobilenetv2_v2_aug_medium [tflite_float16] | 97 | 19 | 8.724e-13 |
| app_yolo11n-cls_v2 [tflite] | mobilenetv3large_v2_aug_medium [tflite_float16] | 64 | 28 | 0.0002633 |
| app_yolo11n-cls_v2 [tflite] | mobilenetv3large_v2_aug_medium [keras] | 65 | 28 | 0.0001892 |
| app_yolo11n-cls_v2 [tflite] | legacy_v1_efficientnet_lite0 [tflite_float16] | 66 | 25 | 2.751e-05 |
| app_yolo11n-cls_v2 [tflite] | efficientnetb0_v2_aug_medium [keras] | 75 | 26 | 1.787e-06 |
| app_yolo11n-cls_v2 [tflite] | efficientnetb0_v2_aug_medium [tflite_float16] | 75 | 26 | 1.787e-06 |
| app_yolo11n-cls_v2 [tflite] | legacy_v1_mobilenetv2_finetuned [keras] | 74 | 19 | 2.149e-08 |
| app_yolo11n-cls_v2 [tflite] | mobilenetv2_v2_aug_medium [keras] | 95 | 24 | 1.391e-10 |
| app_yolo11n-cls_v2 [tflite] | mobilenetv2_v2_aug_medium [tflite_float16] | 97 | 24 | 5.931e-11 |
| mobilenetv3large_v2_aug_medium [tflite_float16] | mobilenetv3large_v2_aug_medium [keras] | 1 | 0 | 1 |
| mobilenetv3large_v2_aug_medium [tflite_float16] | legacy_v1_efficientnet_lite0 [tflite_float16] | 52 | 48 | 0.7642 |
| mobilenetv3large_v2_aug_medium [tflite_float16] | efficientnetb0_v2_aug_medium [keras] | 47 | 34 | 0.1824 |
| mobilenetv3large_v2_aug_medium [tflite_float16] | efficientnetb0_v2_aug_medium [tflite_float16] | 47 | 34 | 0.1824 |
| mobilenetv3large_v2_aug_medium [tflite_float16] | legacy_v1_mobilenetv2_finetuned [keras] | 57 | 39 | 0.08273 |
| mobilenetv3large_v2_aug_medium [tflite_float16] | mobilenetv2_v2_aug_medium [keras] | 71 | 36 | 0.001013 |
| mobilenetv3large_v2_aug_medium [tflite_float16] | mobilenetv2_v2_aug_medium [tflite_float16] | 73 | 36 | 0.0005644 |
| mobilenetv3large_v2_aug_medium [keras] | legacy_v1_efficientnet_lite0 [tflite_float16] | 51 | 48 | 0.8407 |
| mobilenetv3large_v2_aug_medium [keras] | efficientnetb0_v2_aug_medium [keras] | 46 | 34 | 0.2188 |
| mobilenetv3large_v2_aug_medium [keras] | efficientnetb0_v2_aug_medium [tflite_float16] | 46 | 34 | 0.2188 |
| mobilenetv3large_v2_aug_medium [keras] | legacy_v1_mobilenetv2_finetuned [keras] | 57 | 40 | 0.1043 |
| mobilenetv3large_v2_aug_medium [keras] | mobilenetv2_v2_aug_medium [keras] | 71 | 37 | 0.001496 |
| mobilenetv3large_v2_aug_medium [keras] | mobilenetv2_v2_aug_medium [tflite_float16] | 73 | 37 | 0.0008465 |
| legacy_v1_efficientnet_lite0 [tflite_float16] | efficientnetb0_v2_aug_medium [keras] | 47 | 39 | 0.4504 |
| legacy_v1_efficientnet_lite0 [tflite_float16] | efficientnetb0_v2_aug_medium [tflite_float16] | 47 | 39 | 0.4504 |
| legacy_v1_efficientnet_lite0 [tflite_float16] | legacy_v1_mobilenetv2_finetuned [keras] | 53 | 40 | 0.2134 |
| legacy_v1_efficientnet_lite0 [tflite_float16] | mobilenetv2_v2_aug_medium [keras] | 67 | 43 | 0.02831 |
| legacy_v1_efficientnet_lite0 [tflite_float16] | mobilenetv2_v2_aug_medium [tflite_float16] | 69 | 43 | 0.01816 |
| efficientnetb0_v2_aug_medium [keras] | efficientnetb0_v2_aug_medium [tflite_float16] | 0 | 0 | 1 |
| efficientnetb0_v2_aug_medium [keras] | legacy_v1_mobilenetv2_finetuned [keras] | 55 | 49 | 0.6239 |
| efficientnetb0_v2_aug_medium [keras] | mobilenetv2_v2_aug_medium [keras] | 64 | 42 | 0.04138 |
| efficientnetb0_v2_aug_medium [keras] | mobilenetv2_v2_aug_medium [tflite_float16] | 65 | 41 | 0.02549 |
| efficientnetb0_v2_aug_medium [tflite_float16] | legacy_v1_mobilenetv2_finetuned [keras] | 55 | 49 | 0.6239 |
| efficientnetb0_v2_aug_medium [tflite_float16] | mobilenetv2_v2_aug_medium [keras] | 64 | 42 | 0.04138 |
| efficientnetb0_v2_aug_medium [tflite_float16] | mobilenetv2_v2_aug_medium [tflite_float16] | 65 | 41 | 0.02549 |
| legacy_v1_mobilenetv2_finetuned [keras] | mobilenetv2_v2_aug_medium [keras] | 51 | 41 | 0.3481 |
| legacy_v1_mobilenetv2_finetuned [keras] | mobilenetv2_v2_aug_medium [tflite_float16] | 53 | 41 | 0.2566 |
| mobilenetv2_v2_aug_medium [keras] | mobilenetv2_v2_aug_medium [tflite_float16] | 3 | 1 | 0.625 |
