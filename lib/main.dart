import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'services/app_services.dart';
import 'services/fact_service.dart';
import 'services/yolo_classifier.dart';
import 'theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ChemSortApp());
}

class ChemSortApp extends StatefulWidget {
  const ChemSortApp({super.key});

  @override
  State<ChemSortApp> createState() => _ChemSortAppState();
}

class _ChemSortAppState extends State<ChemSortApp> {
  late final Future<(YoloClassifier, FactService)> _services =
      AppServices.load();

  @override
  Widget build(BuildContext context) {
    // AppServices sits above MaterialApp so every pushed route can reach it.
    return FutureBuilder(
      future: _services,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return _app(_StartupError(error: snapshot.error!));
        }
        if (!snapshot.hasData) {
          return _app(const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          ));
        }
        final (classifier, facts) = snapshot.data!;
        return AppServices(
          classifier: classifier,
          facts: facts,
          child: _app(const HomeScreen()),
        );
      },
    );
  }

  Widget _app(Widget home) => MaterialApp(
        title: 'ChemSort',
        debugShowCheckedModeBanner: false,
        theme: buildTheme(),
        home: home,
      );
}

class _StartupError extends StatelessWidget {
  const _StartupError({required this.error});

  final Object error;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 48, color: Colors.red),
              const SizedBox(height: 16),
              const Text(
                'Không khởi động được: không tải được model nhận diện hoặc dữ liệu flashcard.',
                textAlign: TextAlign.center,
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 8),
              Text('$error', textAlign: TextAlign.center),
            ],
          ),
        ),
      ),
    );
  }
}
