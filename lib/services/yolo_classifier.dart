import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/prediction.dart';
import 'mapping_service.dart';

const _inputSize = 224;

/// Runs the YOLO11n-cls TFLite model on the phone.
///
/// The test-set scores in the README were measured with exactly the
/// preprocessing in [preprocessImage] below.
class YoloClassifier {
  YoloClassifier._(this._interpreter, this._mapping, this._channelsFirst);

  static Future<YoloClassifier> load({
    required String modelAsset,
    required MappingService mapping,
  }) async {
    final interpreter = await Interpreter.fromAsset(
      modelAsset,
      options: InterpreterOptions()..threads = 4,
    );
    final channelsFirst = _checkTensors(interpreter, mapping.modelLabels.length);
    return YoloClassifier._(interpreter, mapping, channelsFirst);
  }

  final Interpreter _interpreter;
  final MappingService _mapping;

  /// LiteRT exports keep PyTorch's NCHW input; onnx2tf exports are NHWC.
  final bool _channelsFirst;

  /// Returns whether the input is channels-first (NCHW).
  static bool _checkTensors(Interpreter interpreter, int classCount) {
    final input = interpreter.getInputTensor(0);
    final output = interpreter.getOutputTensor(0);
    final channelsFirst = listEquals(input.shape, const [1, 3, _inputSize, _inputSize]);
    if (!(channelsFirst || listEquals(input.shape, const [1, _inputSize, _inputSize, 3])) ||
        input.type != TensorType.float32) {
      throw StateError('Unexpected input tensor: ${input.shape} ${input.type}');
    }
    if (!listEquals(output.shape, [1, classCount]) ||
        output.type != TensorType.float32) {
      throw StateError('Unexpected output tensor: ${output.shape} ${output.type}');
    }
    return channelsFirst;
  }

  /// The exported model already ends in softmax, so the output is used as is.
  Future<Prediction> classify(Uint8List imageBytes) async {
    var input = await compute(preprocessImage, imageBytes);
    if (_channelsFirst) input = toChannelsFirst(input);
    final output = [List<double>.filled(_mapping.modelLabels.length, 0)];
    _interpreter.run(input.buffer.asUint8List(), output);
    return _mapping.predict(output.first);
  }

  void close() => _interpreter.close();
}

/// EXIF orientation, RGB, centre square crop, area-average resize to
/// 224x224, pixels scaled to 0..1 (YOLO-cls has no mean/std normalisation).
///
/// Ultralytics validates with resize-shorter-side + centre crop; the crop
/// first and an area resize give the same framing without aliasing.
Float32List preprocessImage(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw const FormatException('Không đọc được ảnh');
  }
  final oriented = img.bakeOrientation(decoded);
  final side = oriented.width < oriented.height ? oriented.width : oriented.height;
  final square = img
      .copyCrop(
        oriented,
        x: (oriented.width - side) ~/ 2,
        y: (oriented.height - side) ~/ 2,
        width: side,
        height: side,
      )
      .convert(format: img.Format.uint8, numChannels: 3);
  return boxResize(square.getBytes(order: img.ChannelOrder.rgb), side);
}

/// HWC (interleaved RGB) to CHW (one 224x224 plane per channel).
Float32List toChannelsFirst(Float32List hwc) {
  const plane = _inputSize * _inputSize;
  final chw = Float32List(hwc.length);
  for (var i = 0; i < plane; i++) {
    chw[i] = hwc[i * 3];
    chw[plane + i] = hwc[i * 3 + 1];
    chw[2 * plane + i] = hwc[i * 3 + 2];
  }
  return chw;
}

/// Area-average resize of a square RGB image to 224x224, scaled to 0..1.
/// Upscaling falls back to nearest neighbour.
Float32List boxResize(Uint8List rgb, int side) {
  final tensor = Float32List(_inputSize * _inputSize * 3);
  if (side <= _inputSize) {
    var i = 0;
    for (var y = 0; y < _inputSize; y++) {
      final row = (y * side ~/ _inputSize) * side;
      for (var x = 0; x < _inputSize; x++) {
        final p = (row + x * side ~/ _inputSize) * 3;
        for (var c = 0; c < 3; c++) {
          tensor[i++] = rgb[p + c] / 255;
        }
      }
    }
    return tensor;
  }

  // Sum each output block directly; every input pixel is read once.
  final edges = [for (var i = 0; i <= _inputSize; i++) i * side ~/ _inputSize];
  final rowSums = Float64List(_inputSize * 3);
  for (var oy = 0; oy < _inputSize; oy++) {
    rowSums.fillRange(0, rowSums.length, 0);
    for (var y = edges[oy]; y < edges[oy + 1]; y++) {
      final row = y * side;
      for (var ox = 0; ox < _inputSize; ox++) {
        for (var x = edges[ox]; x < edges[ox + 1]; x++) {
          final p = (row + x) * 3;
          rowSums[ox * 3] += rgb[p];
          rowSums[ox * 3 + 1] += rgb[p + 1];
          rowSums[ox * 3 + 2] += rgb[p + 2];
        }
      }
    }
    final blockHeight = edges[oy + 1] - edges[oy];
    for (var ox = 0; ox < _inputSize; ox++) {
      final area = blockHeight * (edges[ox + 1] - edges[ox]) * 255.0;
      final out = (oy * _inputSize + ox) * 3;
      for (var c = 0; c < 3; c++) {
        tensor[out + c] = rowSums[ox * 3 + c] / area;
      }
    }
  }
  return tensor;
}
