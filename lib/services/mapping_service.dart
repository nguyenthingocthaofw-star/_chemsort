import 'dart:convert';

import '../models/prediction.dart';
import '../models/waste_material.dart';

/// Aggregates the 9 RealWaste probabilities into the 6 app classes.
///
/// Probabilities inside a group are summed before argmax, the same rule
/// used when the model was evaluated (see README, "Đánh giá").
class MappingService {
  MappingService._(this.modelLabels, this._groups);

  /// Builds the service from `labels_9.txt` and `chemsort_mapping.json`,
  /// refusing any bundle whose labels and mapping disagree.
  factory MappingService.fromAssets({
    required String labelsText,
    required String mappingJson,
  }) {
    final labels = const LineSplitter()
        .convert(labelsText.replaceFirst('﻿', ''))
        .map((line) => line.trim())
        .where((line) => line.isNotEmpty)
        .toList(growable: false);
    final mapping = jsonDecode(mappingJson) as Map<String, dynamic>;

    final mappingOrder = List<String>.from(
      mapping['model_output_9class']['classes_in_index_order'] as List,
    );
    if (!_sameOrder(labels, mappingOrder)) {
      throw const FormatException(
        'labels_9.txt and chemsort_mapping.json disagree on class order',
      );
    }

    final appOutput = mapping['app_output_6class'] as Map<String, dynamic>;
    final appOrder = List<String>.from(appOutput['classes_in_index_order'] as List);
    final groupsJson = appOutput['groups'] as Map<String, dynamic>;
    final indexByLabel = {for (var i = 0; i < labels.length; i++) labels[i]: i};

    final groups = <WasteMaterial, List<int>>{};
    final covered = <int>{};
    for (final appClass in appOrder) {
      final indices = (groupsJson[appClass] as List).map((label) {
        final index = indexByLabel[label];
        if (index == null) {
          throw FormatException('Mapping references unknown label: $label');
        }
        if (!covered.add(index)) {
          throw FormatException('Label mapped more than once: $label');
        }
        return index;
      }).toList(growable: false);
      groups[WasteMaterial.fromAppClass(appClass)] = indices;
    }
    if (covered.length != labels.length) {
      throw const FormatException('Mapping does not cover every model label');
    }
    return MappingService._(labels, groups);
  }

  final List<String> modelLabels;
  final Map<WasteMaterial, List<int>> _groups;

  Prediction predict(List<double> probabilities9) {
    if (probabilities9.length != modelLabels.length) {
      throw ArgumentError(
        'Expected ${modelLabels.length} probabilities, '
        'got ${probabilities9.length}',
      );
    }
    final probabilities6 = {
      for (final entry in _groups.entries)
        entry.key: entry.value.fold<double>(0, (sum, i) => sum + probabilities9[i]),
    };
    final best = probabilities6.entries
        .reduce((a, b) => b.value > a.value ? b : a);
    final modelIndex = _argmax(probabilities9);
    return Prediction(
      material: best.key,
      confidence: best.value,
      modelClass: modelLabels[modelIndex],
      modelConfidence: probabilities9[modelIndex],
      probabilities6: probabilities6,
    );
  }

  static int _argmax(List<double> values) {
    var best = 0;
    for (var i = 1; i < values.length; i++) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }

  static bool _sameOrder(List<String> a, List<String> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
