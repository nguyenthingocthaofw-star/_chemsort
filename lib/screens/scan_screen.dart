import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../config.dart';
import '../services/app_services.dart';
import '../theme.dart';
import 'result_screen.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  /// The flashcard screen pops back here, so a new fact needs a new scan.
  static const routeName = '/scan';

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  final _picker = ImagePicker();
  Uint8List? _imageBytes;
  bool _busy = false;
  String? _message;

  Future<void> _pick(ImageSource source) async {
    if (_busy) return;

    final classifier = AppServices.of(context).classifier;
    setState(() {
      _busy = true;
      _message = null;
    });

    try {
      // Downscale on the platform side; the model only needs 224x224.
      final file = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 95,
        requestFullMetadata: false,
      );
      if (file == null) return;

      final bytes = await file.readAsBytes();
      if (!mounted) return;
      setState(() => _imageBytes = bytes);

      final prediction = await classifier.classify(bytes);
      if (!mounted) return;

      if (prediction.confidence < AppConfig.confidenceThreshold) {
        setState(() => _message =
            'Chưa nhận diện chắc chắn '
            '(${(prediction.confidence * 100).toStringAsFixed(0)}%). '
            'Hãy chụp gần hơn, đủ sáng và chỉ một vật thể.');
        return;
      }

      await Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => ResultScreen(imageBytes: bytes, prediction: prediction),
      ));
      // Back from the result/fact: the previous photo is used up.
      if (mounted) setState(() => _imageBytes = null);
    } catch (error) {
      if (!mounted) return;
      setState(() => _message = source == ImageSource.camera
          ? 'Không chụp hoặc phân tích được ảnh: $error'
          : 'Không mở hoặc phân tích được ảnh: $error');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Quét vật thể')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: chemGreen, width: 2),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      if (_imageBytes != null)
                        Image.memory(_imageBytes!, fit: BoxFit.cover)
                      else
                        const Center(
                          child: Padding(
                            padding: EdgeInsets.all(24),
                            child: Text(
                              'Chụp hoặc chọn ảnh\nmột vật thể rác',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: Colors.black87,
                                fontSize: 26,
                                fontWeight: FontWeight.w700,
                                height: 1.3,
                              ),
                            ),
                          ),
                        ),
                      if (_busy)
                        const ColoredBox(
                          color: Colors.black45,
                          child: Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                CircularProgressIndicator(color: Colors.white),
                                SizedBox(height: 12),
                                Text('Đang nhận diện…',
                                    style: TextStyle(color: Colors.white)),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
              if (_message != null) ...[
                const SizedBox(height: 16),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.orange.shade50,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(_message!,
                      style: TextStyle(color: Colors.orange.shade900)),
                ),
              ],
              const SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: FilledButton.icon(
                      onPressed:
                          _busy ? null : () => _pick(ImageSource.camera),
                      icon: const Icon(Icons.photo_camera),
                      label: const Text('Chụp ảnh'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed:
                          _busy ? null : () => _pick(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library_outlined,
                          color: chemGreen),
                      label: const Text('Album'),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
