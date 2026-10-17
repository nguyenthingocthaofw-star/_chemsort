/// Asset paths and tunables shared by the services.
abstract final class AppConfig {
  static const assetsDir = 'assets/chemsort';
  static const flashcardsAsset = '$assetsDir/flashcards.json';
  static const modelAsset = '$assetsDir/yolo11n-cls_data_v2_float32.tflite';
  static const labelsAsset = '$assetsDir/labels_9.txt';
  static const mappingAsset = '$assetsDir/chemsort_mapping.json';

  /// Minimum summed softmax of the winning 6-class group before a result
  /// is shown.
  static const confidenceThreshold = 0.5;
}
