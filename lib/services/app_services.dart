import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

import '../config.dart';
import 'fact_service.dart';
import 'mapping_service.dart';
import 'yolo_classifier.dart';

class AppServices extends InheritedWidget {
  const AppServices({
    super.key,
    required this.classifier,
    required this.facts,
    required super.child,
  });

  final YoloClassifier classifier;
  final FactService facts;

  static Future<(YoloClassifier, FactService)> load() async {
    final mapping = MappingService.fromAssets(
      labelsText: await rootBundle.loadString(AppConfig.labelsAsset),
      mappingJson: await rootBundle.loadString(AppConfig.mappingAsset),
    );
    final classifier = await YoloClassifier.load(
      modelAsset: AppConfig.modelAsset,
      mapping: mapping,
    );
    final facts = FactService.fromJson(
      await rootBundle.loadString(AppConfig.flashcardsAsset),
    );
    return (classifier, facts);
  }

  static AppServices of(BuildContext context) {
    final services = context.dependOnInheritedWidgetOfExactType<AppServices>();
    assert(services != null, 'AppServices not found in the widget tree');
    return services!;
  }

  @override
  bool updateShouldNotify(AppServices oldWidget) =>
      classifier != oldWidget.classifier || facts != oldWidget.facts;
}
