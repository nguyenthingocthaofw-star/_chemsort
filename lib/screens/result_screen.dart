import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../models/prediction.dart';
import '../models/waste_material.dart';
import '../theme.dart';
import '../widgets/group_badge.dart';
import 'flashcard_screen.dart';
import 'resin_screen.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({
    super.key,
    required this.imageBytes,
    required this.prediction,
  });

  final Uint8List imageBytes;
  final Prediction prediction;

  void _continue(BuildContext context) {
    final material = prediction.material;
    if (material == WasteMaterial.plastic) {
      Navigator.of(context)
          .push(MaterialPageRoute(builder: (_) => const ResinScreen()));
    } else {
      showFactForScan(context, material);
    }
  }

  @override
  Widget build(BuildContext context) {
    final material = prediction.material;
    final textTheme = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('Kết quả nhận diện')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text(
              material.titleEn,
              textAlign: TextAlign.center,
              style: textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.w800,
                letterSpacing: 1,
              ),
            ),
            Text(
              material.titleVi,
              textAlign: TextAlign.center,
              style: textTheme.titleMedium?.copyWith(color: Colors.black54),
            ),
            const SizedBox(height: 16),
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: AspectRatio(
                aspectRatio: 4 / 3,
                child: Image.memory(imageBytes, fit: BoxFit.cover),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Độ tin cậy: ${(prediction.confidence * 100).toStringAsFixed(1)}%',
              textAlign: TextAlign.center,
              style: textTheme.bodySmall?.copyWith(color: Colors.black54),
            ),
            const SizedBox(height: 16),
            GroupBadge(material: material),
            const SizedBox(height: 16),
            Text(material.hint, style: textTheme.bodyLarge),
            const SizedBox(height: 12),
            Text(
              'Lưu ý: việc xếp nhóm dựa trên phân loại chung tại Việt Nam, '
              'không đồng nghĩa vật liệu chắc chắn được tái chế tại mọi địa phương.',
              style: textTheme.bodySmall?.copyWith(color: Colors.black45),
            ),
            const SizedBox(height: 24),
            FilledButton(
              onPressed: () => _continue(context),
              child: const Text('Tiếp tục'),
            ),
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              style: TextButton.styleFrom(foregroundColor: chemGreen),
              child: const Text('Quét lại'),
            ),
          ],
        ),
      ),
    );
  }
}
