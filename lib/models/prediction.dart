import 'waste_material.dart';

/// One classification result, already aggregated to the six app materials.
class Prediction {
  const Prediction({
    required this.material,
    required this.confidence,
    required this.modelClass,
    required this.modelConfidence,
    required this.probabilities6,
  });

  final WasteMaterial material;

  /// Summed probability of the winning 6-class group.
  final double confidence;

  /// Best raw 9-class RealWaste label, useful for debugging.
  final String modelClass;
  final double modelConfidence;
  final Map<WasteMaterial, double> probabilities6;
}
