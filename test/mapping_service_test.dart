import 'dart:convert';

import 'package:chemsort/models/waste_material.dart';
import 'package:chemsort/services/mapping_service.dart';
import 'package:flutter_test/flutter_test.dart';

const _labels = [
  'Cardboard',
  'Food Organics',
  'Glass',
  'Metal',
  'Miscellaneous Trash',
  'Paper',
  'Plastic',
  'Textile Trash',
  'Vegetation',
];

String _mappingJson({List<String> order = _labels}) => jsonEncode({
      'model_output_9class': {'classes_in_index_order': order},
      'app_output_6class': {
        'classes_in_index_order': [
          'Paper/Cardboard',
          'Organic',
          'Glass',
          'Metal',
          'Other',
          'Plastic',
        ],
        'groups': {
          'Paper/Cardboard': ['Cardboard', 'Paper'],
          'Organic': ['Food Organics'],
          'Glass': ['Glass'],
          'Metal': ['Metal'],
          'Other': ['Miscellaneous Trash', 'Textile Trash', 'Vegetation'],
          'Plastic': ['Plastic'],
        },
      },
    });

MappingService _service() => MappingService.fromAssets(
      labelsText: '﻿${_labels.join('\n')}\n',
      mappingJson: _mappingJson(),
    );

void main() {
  test('sums Paper and Cardboard before argmax', () {
    // Plastic is the single best raw class, but Paper+Cardboard wins.
    final prediction = _service()
        .predict([0.25, 0, 0, 0, 0, 0.20, 0.30, 0.10, 0.15]);

    expect(prediction.material, WasteMaterial.paperCardboard);
    expect(prediction.confidence, closeTo(0.45, 1e-9));
    expect(prediction.modelClass, 'Plastic');
  });

  test('sums the three Other sources', () {
    final prediction = _service()
        .predict([0.1, 0.1, 0, 0, 0.2, 0, 0.25, 0.15, 0.2]);

    expect(prediction.material, WasteMaterial.other);
    expect(prediction.probabilities6[WasteMaterial.other], closeTo(0.55, 1e-9));
    expect(prediction.material.vnGroup, isNull);
  });

  test('rejects labels that disagree with the mapping order', () {
    expect(
      () => MappingService.fromAssets(
        labelsText: _labels.reversed.join('\n'),
        mappingJson: _mappingJson(),
      ),
      throwsFormatException,
    );
  });

  test('rejects a probability vector of the wrong length', () {
    expect(() => _service().predict([1, 0, 0]), throwsArgumentError);
  });
}
