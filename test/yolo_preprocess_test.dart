import 'dart:typed_data';

import 'package:chemsort/services/yolo_classifier.dart';
import 'package:flutter_test/flutter_test.dart';

Uint8List _square(int side, int Function(int x, int y) value) {
  final rgb = Uint8List(side * side * 3);
  for (var y = 0; y < side; y++) {
    for (var x = 0; x < side; x++) {
      rgb.fillRange((y * side + x) * 3, (y * side + x) * 3 + 3, value(x, y));
    }
  }
  return rgb;
}

void main() {
  test('224x224 input is only scaled to 0..1', () {
    final rgb = Uint8List.fromList(List.generate(224 * 224 * 3, (i) => i % 251));
    final tensor = boxResize(rgb, 224);
    expect(tensor, orderedEquals(rgb.map((v) => closeTo(v / 255, 1e-6))));
  });

  test('a 2x downscale averages each 2x2 block', () {
    final tensor = boxResize(_square(448, (x, y) => x.isEven ? 0 : 200), 448);
    expect(tensor.every((v) => (v - 100 / 255).abs() < 1e-6), isTrue);
  });

  test('a non-integer downscale keeps a ramp monotonic and in range', () {
    final tensor = boxResize(_square(1000, (x, _) => x * 255 ~/ 999), 1000);
    for (var x = 1; x < 224; x++) {
      expect(tensor[x * 3], greaterThanOrEqualTo(tensor[(x - 1) * 3]));
    }
    expect(tensor.every((v) => v >= 0 && v <= 1), isTrue);
  });

  test('channels-first puts each colour in its own plane', () {
    final hwc = Float32List(224 * 224 * 3);
    for (var i = 0; i < 224 * 224; i++) {
      hwc.setAll(i * 3, [0.1, 0.5, 0.9]);
    }
    final chw = toChannelsFirst(hwc);
    const plane = 224 * 224;
    expect(chw.sublist(0, plane).every((v) => v == hwc[0]), isTrue);
    expect(chw.sublist(plane, 2 * plane).every((v) => v == hwc[1]), isTrue);
    expect(chw.sublist(2 * plane).every((v) => v == hwc[2]), isTrue);
  });
}
