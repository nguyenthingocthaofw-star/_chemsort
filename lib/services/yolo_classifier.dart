import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/prediction.dart';
import '../models/waste_material.dart';
import 'mapping_service.dart';

const _inputSize = 224;

/// Configuration for adaptive grid-based classification
abstract final class ClassifierConfig {
  /// Enable multi-region classification for complex images
  static const enableMultiRegion = true;

  /// Quick path threshold: confidence above this uses single inference
  static const quickThreshold = 0.70; // Giảm từ 0.75 để dùng grid nhiều hơn

  /// Early stopping threshold: stop grid search when confidence exceeds this
  static const earlyStopThreshold = 0.85;

  /// Good enough threshold: acceptable confidence after checking half grids
  static const goodEnoughThreshold = 0.65; // Giảm từ 0.70

  /// Maximum number of grids to prevent excessive inference
  static const maxGrids = 10;

  /// Enable debug logging
  static const enableLogging = true; // Bật để debug thủy tinh

  /// Material-specific thresholds for quick path
  /// Thủy tinh và nhựa trong suốt cần threshold thấp hơn
  static const Map<WasteMaterial, double> materialThresholds = {
    WasteMaterial.glass: 0.60, // Thủy tinh trong suốt
    WasteMaterial.plastic: 0.65, // Nhựa trong suốt
    WasteMaterial.paperCardboard: 0.70,
    WasteMaterial.metal: 0.70,
    WasteMaterial.organic: 0.70,
    WasteMaterial.other: 0.50, // "Khác" dễ accept hơn
  };

  /// Transparency detection threshold
  static const transparencyThreshold = 0.6; // >60% low saturation = transparent

  /// Enhanced contrast for transparent objects
  static const transparentContrast = 1.30; // Tăng từ 1.15
  static const transparentBrightness = 1.10;
  static const transparentSaturation = 1.20;
}

/// Grid configuration based on image aspect ratio
class _GridConfig {
  final int rows;
  final int cols;
  final int totalGrids;

  _GridConfig({
    required this.rows,
    required this.cols,
  }) : totalGrids = rows * cols;

  /// Priority order: Center → Corners → Edges
  List<int> get priorityOrder {
    final center = (rows ~/ 2) * cols + (cols ~/ 2);
    final corners = [
      0, // Top-left
      cols - 1, // Top-right
      (rows - 1) * cols, // Bottom-left
      rows * cols - 1, // Bottom-right
    ].where((i) => i >= 0 && i < totalGrids).toList();

    final all = List.generate(totalGrids, (i) => i);
    final edges =
        all.where((i) => i != center && !corners.contains(i)).toList();

    return [center, ...corners, ...edges];
  }
}

/// Image complexity level for adaptive processing
enum _ImageComplexity { simple, moderate, complex }

/// Result from a single grid classification
class _GridResult {
  final int gridIndex;
  final Prediction prediction;

  _GridResult({
    required this.gridIndex,
    required this.prediction,
  });
}

/// Runs the YOLO11n-cls TFLite model on the phone.
///
/// Supports adaptive multi-region classification for improved accuracy
/// with complex images while maintaining fast inference for simple cases.
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
    final channelsFirst =
        _checkTensors(interpreter, mapping.modelLabels.length);
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
    final channelsFirst =
        listEquals(input.shape, const [1, 3, _inputSize, _inputSize]);
    if (!(channelsFirst ||
            listEquals(input.shape, const [1, _inputSize, _inputSize, 3])) ||
        input.type != TensorType.float32) {
      throw StateError('Unexpected input tensor: ${input.shape} ${input.type}');
    }
    if (!listEquals(output.shape, [1, classCount]) ||
        output.type != TensorType.float32) {
      throw StateError(
          'Unexpected output tensor: ${output.shape} ${output.type}');
    }
    return channelsFirst;
  }

  /// Main classification entry point with adaptive multi-region support.
  ///
  /// Strategy:
  /// 1. Quick path (70% cases): Single center crop, ~50ms
  /// 2. Saliency path (20% cases): Edge detection + smart crop, ~80ms
  /// 3. Grid path (10% cases): Adaptive grid with early stopping, ~150ms
  Future<Prediction> classify(Uint8List imageBytes) async {
    if (!ClassifierConfig.enableMultiRegion) {
      // Fallback to original implementation
      return _classifyQuick(imageBytes);
    }

    final startTime = DateTime.now();

    // Step 1: Quick classification with center crop
    final quickResult = await _classifyQuick(imageBytes);

    // ✅ Check threshold dựa trên material type
    final threshold =
        ClassifierConfig.materialThresholds[quickResult.material] ??
            ClassifierConfig.quickThreshold;

    if (quickResult.confidence > threshold) {
      _logResult('Quick path', quickResult, startTime, 1);
      return quickResult;
    }

    // Step 2: Decode image for advanced processing
    final decoded = await compute(_decodeImage, imageBytes);
    if (decoded == null) {
      throw const FormatException('Không đọc được ảnh');
    }

    // Step 3: Analyze image complexity
    final complexity = _analyzeImageComplexity(decoded);

    Prediction result;
    int gridsProcessed;

    switch (complexity) {
      case _ImageComplexity.simple:
        // Use saliency detection for simple images
        result = await _classifySaliency(decoded);
        gridsProcessed = 1;
        _logResult('Saliency path', result, startTime, gridsProcessed);
        break;

      case _ImageComplexity.moderate:
        // Use 5-grid smart layout
        result = await _classifySmartGrids(decoded, gridCount: 5);
        gridsProcessed = 5;
        _logResult('5-grid path', result, startTime, gridsProcessed);
        break;

      case _ImageComplexity.complex:
        // Use full adaptive grid with early stopping
        final gridResult = await _classifyAdaptiveGrid(decoded);
        result = gridResult.$1;
        gridsProcessed = gridResult.$2;
        _logResult('Adaptive grid path', result, startTime, gridsProcessed);
        break;
    }

    return result;
  }

  /// Quick classification using center crop (original method)
  Future<Prediction> _classifyQuick(Uint8List imageBytes) async {
    var input = await compute(preprocessImage, imageBytes);
    if (_channelsFirst) input = toChannelsFirst(input);
    final output = [List<double>.filled(_mapping.modelLabels.length, 0)];
    _interpreter.run(input.buffer.asUint8List(), output);
    return _mapping.predict(output.first);
  }

  /// Classify using saliency detection (edge-based smart crop)
  Future<Prediction> _classifySaliency(img.Image decoded) async {
    // ✅ Detect if image contains transparent objects
    final isTransparent = _isTransparent(decoded);

    // ✅ Apply enhanced contrast for transparent objects
    final enhanced = isTransparent
        ? img.adjustColor(
            decoded,
            contrast: ClassifierConfig.transparentContrast,
            brightness: ClassifierConfig.transparentBrightness,
            saturation: ClassifierConfig.transparentSaturation,
          )
        : img.adjustColor(
            decoded,
            contrast: 1.15,
            brightness: 1.05,
          );

    // Try to find salient region
    final salientRegion = await compute(_findSalientRegion, enhanced);

    final imageToClassify = salientRegion ?? enhanced;
    final bytes = Uint8List.fromList(img.encodeJpg(imageToClassify));

    return _classifyQuick(bytes);
  }

  /// Classify using smart grid layout (5 regions: 4 corners + center)
  Future<Prediction> _classifySmartGrids(img.Image decoded,
      {required int gridCount}) async {
    final enhanced = img.adjustColor(decoded, contrast: 1.15);
    final grids = _generate5Grids(enhanced);

    var bestPrediction = await _classifyImageDirect(grids[0]);

    for (var i = 1; i < grids.length && i < gridCount; i++) {
      final prediction = await _classifyImageDirect(grids[i]);

      if (prediction.confidence > bestPrediction.confidence) {
        bestPrediction = prediction;
      }

      // Early stop if confidence is high enough
      if (bestPrediction.confidence >= ClassifierConfig.earlyStopThreshold) {
        break;
      }
    }

    return bestPrediction;
  }

  /// Classify using adaptive grid with early stopping
  /// Returns (prediction, gridsProcessed)
  Future<(Prediction, int)> _classifyAdaptiveGrid(img.Image decoded) async {
    final enhanced = img.adjustColor(decoded, contrast: 1.15);
    final config = _calculateGridLayout(enhanced);
    final grids = _generateAdaptiveGrids(enhanced, config);

    final gridResults = <_GridResult>[];
    var bestPrediction = Prediction(
      material: WasteMaterial.other,
      confidence: 0.0,
      modelClass: '',
      modelConfidence: 0.0,
      probabilities6: {},
    );

    final priorityOrder = config.priorityOrder;

    for (var i = 0; i < priorityOrder.length; i++) {
      final index = priorityOrder[i];
      final prediction = await _classifyImageDirect(grids[index]);

      gridResults.add(_GridResult(
        gridIndex: index,
        prediction: prediction,
      ));

      if (prediction.confidence > bestPrediction.confidence) {
        bestPrediction = prediction;
      }

      // Early stop conditions
      if (bestPrediction.confidence >= ClassifierConfig.earlyStopThreshold) {
        return (_selectBestFromGrids(gridResults), gridResults.length);
      }

      // Good enough stop after checking half the grids
      if (bestPrediction.confidence >= ClassifierConfig.goodEnoughThreshold &&
          gridResults.length >= grids.length ~/ 2) {
        return (_selectBestFromGrids(gridResults), gridResults.length);
      }
    }

    return (_selectBestFromGrids(gridResults), gridResults.length);
  }

  /// Classify an image directly (from img.Image)
  Future<Prediction> _classifyImageDirect(img.Image image) async {
    final bytes = Uint8List.fromList(img.encodeJpg(image));
    return _classifyQuick(bytes);
  }

  void _logResult(
      String path, Prediction result, DateTime startTime, int grids) {
    if (ClassifierConfig.enableLogging) {
      final latency = DateTime.now().difference(startTime).inMilliseconds;
      print('[$path] ${result.material.titleVi} '
          '(conf: ${result.confidence.toStringAsFixed(3)}, '
          'latency: ${latency}ms, grids: $grids)');
    }
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
  final side =
      oriented.width < oriented.height ? oriented.width : oriented.height;
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

// ============================================================================
// HELPER FUNCTIONS - Grid Generation & Selection
// ============================================================================

/// Calculate adaptive grid layout based on image aspect ratio
_GridConfig _calculateGridLayout(img.Image image) {
  final aspectRatio = image.width / image.height;

  _GridConfig config;

  if (aspectRatio >= 2.0) {
    // Ultra-wide (21:9)
    config = _GridConfig(rows: 2, cols: 5);
  } else if (aspectRatio >= 1.5) {
    // Landscape (16:9)
    config = _GridConfig(rows: 2, cols: 4);
  } else if (aspectRatio >= 1.2) {
    // Wide
    config = _GridConfig(rows: 3, cols: 4);
  } else if (aspectRatio >= 0.8) {
    // Square-ish
    config = _GridConfig(rows: 3, cols: 3);
  } else if (aspectRatio >= 0.67) {
    // Slightly tall
    config = _GridConfig(rows: 4, cols: 3);
  } else if (aspectRatio >= 0.5) {
    // Portrait (9:16)
    config = _GridConfig(rows: 4, cols: 2);
  } else {
    // Ultra-tall (9:21)
    config = _GridConfig(rows: 5, cols: 2);
  }

  // Limit max grids to prevent excessive processing
  if (config.totalGrids > ClassifierConfig.maxGrids) {
    if (aspectRatio > 1) {
      // Wide image: reduce columns
      final newCols = (ClassifierConfig.maxGrids / config.rows).floor();
      config = _GridConfig(rows: config.rows, cols: newCols);
    } else {
      // Tall image: reduce rows
      final newRows = (ClassifierConfig.maxGrids / config.cols).floor();
      config = _GridConfig(rows: newRows, cols: config.cols);
    }
  }

  return config;
}

/// Generate adaptive grids with 50% overlap
List<img.Image> _generateAdaptiveGrids(img.Image image, _GridConfig config) {
  final grids = <img.Image>[];

  // Grid size: 60% of image dimension
  final gridWidth = (image.width * 0.6).toInt();
  final gridHeight = (image.height * 0.6).toInt();

  // Step size: 30% (creates 50% overlap)
  final stepX = (image.width * 0.3).toInt();
  final stepY = (image.height * 0.3).toInt();

  for (var row = 0; row < config.rows; row++) {
    for (var col = 0; col < config.cols; col++) {
      var x = col * stepX;
      var y = row * stepY;

      // Clamp to image bounds
      if (x + gridWidth > image.width) {
        x = image.width - gridWidth;
      }
      if (y + gridHeight > image.height) {
        y = image.height - gridHeight;
      }

      x = x.clamp(0, image.width - gridWidth);
      y = y.clamp(0, image.height - gridHeight);

      final grid = img.copyCrop(
        image,
        x: x,
        y: y,
        width: gridWidth,
        height: gridHeight,
      );

      grids.add(grid);
    }
  }

  return grids;
}

/// Generate 5 smart grids: 4 corners + center
List<img.Image> _generate5Grids(img.Image image) {
  final grids = <img.Image>[];
  final size = (image.width * 0.6).toInt().clamp(100, image.width);

  // 4 corners
  grids.add(img.copyCrop(image, x: 0, y: 0, width: size, height: size));

  if (image.width >= size) {
    grids.add(img.copyCrop(image,
        x: image.width - size, y: 0, width: size, height: size));
  }

  if (image.height >= size) {
    grids.add(img.copyCrop(image,
        x: 0, y: image.height - size, width: size, height: size));
  }

  if (image.width >= size && image.height >= size) {
    grids.add(img.copyCrop(image,
        x: image.width - size,
        y: image.height - size,
        width: size,
        height: size));
  }

  // Center
  final centerSize = (image.width * 0.5).toInt().clamp(100, image.width);
  if (image.width >= centerSize && image.height >= centerSize) {
    grids.add(img.copyCrop(image,
        x: (image.width - centerSize) ~/ 2,
        y: (image.height - centerSize) ~/ 2,
        width: centerSize,
        height: centerSize));
  }

  return grids;
}

/// Select best prediction from multiple grid results
Prediction _selectBestFromGrids(List<_GridResult> gridResults) {
  if (gridResults.isEmpty) {
    throw StateError('No grid results');
  }

  // Strategy 1: Find highest confidence
  var best = gridResults.first.prediction;
  for (final result in gridResults) {
    if (result.prediction.confidence > best.confidence) {
      best = result.prediction;
    }
  }

  // Strategy 2: Check for consensus (multiple grids agreeing)
  final materialVotes = <WasteMaterial, List<Prediction>>{};
  for (final result in gridResults) {
    materialVotes
        .putIfAbsent(result.prediction.material, () => [])
        .add(result.prediction);
  }

  // Find material with most votes
  var maxVotes = 0;
  WasteMaterial? consensusMaterial;
  double consensusAvgConf = 0;

  for (final entry in materialVotes.entries) {
    if (entry.value.length > maxVotes) {
      maxVotes = entry.value.length;
      consensusMaterial = entry.key;
      consensusAvgConf =
          entry.value.map((p) => p.confidence).reduce((a, b) => a + b) /
              entry.value.length;
    }
  }

  // Prefer consensus if at least 1/3 grids agree and avg confidence is close
  if (maxVotes >= 3 && consensusAvgConf + 0.1 > best.confidence) {
    final consensusPredictions = materialVotes[consensusMaterial]!;
    best = consensusPredictions
        .reduce((a, b) => a.confidence > b.confidence ? a : b);
  }

  // Strategy 3: Avoid "Other" if there's a better specific option
  if (best.material == WasteMaterial.other && best.confidence < 0.7) {
    final nonMiscPredictions = gridResults
        .map((r) => r.prediction)
        .where((p) => p.material != WasteMaterial.other)
        .toList();

    if (nonMiscPredictions.isNotEmpty) {
      final bestNonMisc = nonMiscPredictions
          .reduce((a, b) => a.confidence > b.confidence ? a : b);

      // Use specific material if confidence is reasonable
      if (bestNonMisc.confidence > 0.5 &&
          bestNonMisc.confidence + 0.15 > best.confidence) {
        best = bestNonMisc;
      }
    }
  }

  return best;
}

// ============================================================================
// IMAGE ANALYSIS FUNCTIONS
// ============================================================================

/// Decode image (isolate-safe function)
img.Image? _decodeImage(Uint8List bytes) {
  return img.decodeImage(bytes);
}

/// Analyze image complexity based on edge density
_ImageComplexity _analyzeImageComplexity(img.Image image) {
  try {
    // Sample image for speed (use 1/4 resolution)
    final sampleWidth = (image.width / 2).toInt();
    final sampleHeight = (image.height / 2).toInt();

    final sampled = img.copyResize(
      image,
      width: sampleWidth,
      height: sampleHeight,
    );

    final gray = img.grayscale(sampled);
    final edges = img.sobel(gray);

    var edgePixels = 0;
    const threshold = 100;

    for (var y = 0; y < edges.height; y++) {
      for (var x = 0; x < edges.width; x++) {
        if (edges.getPixel(x, y).r > threshold) {
          edgePixels++;
        }
      }
    }

    final edgeDensity = edgePixels / (edges.width * edges.height);

    // Classify complexity
    if (edgeDensity < 0.05) {
      return _ImageComplexity.simple;
    } else if (edgeDensity < 0.15) {
      return _ImageComplexity.moderate;
    } else {
      return _ImageComplexity.complex;
    }
  } catch (e) {
    // Fallback to moderate if analysis fails
    return _ImageComplexity.moderate;
  }
}

/// Find salient region using edge detection
img.Image? _findSalientRegion(img.Image image) {
  try {
    final gray = img.grayscale(image);
    final edges = img.sobel(gray);

    // Divide into 6×6 grid to find region with highest edge density
    const gridSize = 6;
    final cellWidth = image.width ~/ gridSize;
    final cellHeight = image.height ~/ gridSize;

    if (cellWidth < 50 || cellHeight < 50) {
      return null; // Image too small
    }

    var maxDensity = 0.0;
    var bestX = 0;
    var bestY = 0;

    // Find 2×2 cell block with highest edge density
    for (var gy = 0; gy < gridSize - 1; gy++) {
      for (var gx = 0; gx < gridSize - 1; gx++) {
        final density = _calculateEdgeDensity(
          edges,
          gx * cellWidth,
          gy * cellHeight,
          cellWidth * 2,
          cellHeight * 2,
        );

        if (density > maxDensity) {
          maxDensity = density;
          bestX = gx * cellWidth;
          bestY = gy * cellHeight;
        }
      }
    }

    // Expand to square and crop
    final regionSize = ((cellWidth * 2 + cellHeight * 2) / 2).toInt();
    final centerX = bestX + cellWidth;
    final centerY = bestY + cellHeight;

    final cropX =
        (centerX - regionSize ~/ 2).clamp(0, image.width - regionSize);
    final cropY =
        (centerY - regionSize ~/ 2).clamp(0, image.height - regionSize);

    return img.copyCrop(
      image,
      x: cropX,
      y: cropY,
      width: regionSize,
      height: regionSize,
    );
  } catch (e) {
    return null; // Fallback if detection fails
  }
}

/// Calculate edge density in a region
double _calculateEdgeDensity(
    img.Image edges, int x, int y, int width, int height) {
  var sum = 0.0;
  var count = 0;

  final maxX = (x + width).clamp(0, edges.width);
  final maxY = (y + height).clamp(0, edges.height);

  for (var py = y; py < maxY; py++) {
    for (var px = x; px < maxX; px++) {
      final pixel = edges.getPixel(px, py);
      sum += pixel.r / 255.0;
      count++;
    }
  }

  return count > 0 ? sum / count : 0;
}

/// Detect if image contains transparent objects (glass, clear plastic)
/// Returns true if >60% of sampled pixels have low saturation + high brightness
bool _isTransparent(img.Image image) {
  try {
    var lowSaturationCount = 0;
    const sampleSize = 100;

    // Sample pixels across the image
    for (var i = 0; i < sampleSize; i++) {
      final x = ((i % 10) * image.width / 10).toInt().clamp(0, image.width - 1);
      final y =
          ((i ~/ 10) * image.height / 10).toInt().clamp(0, image.height - 1);
      final pixel = image.getPixel(x, y);

      // Calculate HSV saturation
      final r = pixel.r / 255.0;
      final g = pixel.g / 255.0;
      final b = pixel.b / 255.0;

      final maxC = [r, g, b].reduce((a, b) => a > b ? a : b);
      final minC = [r, g, b].reduce((a, b) => a < b ? a : b);
      final saturation = maxC == 0 ? 0 : (maxC - minC) / maxC;

      // Low saturation (grayish/colorless) + reasonable brightness = transparent
      if (saturation < 0.2 && r > 0.4) {
        lowSaturationCount++;
      }
    }

    return lowSaturationCount >
        sampleSize * ClassifierConfig.transparencyThreshold;
  } catch (e) {
    return false; // Fallback to non-transparent if detection fails
  }
}
