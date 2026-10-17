import 'package:flutter/material.dart';

import '../models/flashcard.dart';
import '../models/resin.dart';
import '../models/waste_material.dart';
import '../services/app_services.dart';
import '../theme.dart';
import 'scan_screen.dart';

/// Draws the one fact for this scan and shows it. Result/resin screens are
/// removed from the stack, so the only way to another fact is a new scan.
void showFactForScan(
  BuildContext context,
  WasteMaterial material, {
  int? plasticCode,
}) {
  final card = AppServices.of(context)
      .facts
      .nextFact(material, plasticCode: plasticCode);
  Navigator.of(context).pushAndRemoveUntil(
    MaterialPageRoute(
      builder: (_) => FlashcardScreen(
        card: card,
        topic: _topic(material, plasticCode),
      ),
    ),
    (route) => route.settings.name == ScanScreen.routeName || route.isFirst,
  );
}

String _topic(WasteMaterial material, int? plasticCode) {
  if (plasticCode == null) return material.titleVi;
  final resin = Resin.all.firstWhere((r) => r.code == plasticCode);
  return '${resin.code} – ${resin.abbreviation}';
}

/// One verified fact per scan; no "next fact" and no sharing.
class FlashcardScreen extends StatelessWidget {
  const FlashcardScreen({super.key, required this.card, required this.topic});

  final Flashcard card;
  final String topic;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('CHEMISTRY FACT',
            style: TextStyle(color: chemPurple, fontWeight: FontWeight.w700)),
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              Text(topic,
                  style: textTheme.titleSmall?.copyWith(color: Colors.black54)),
              const SizedBox(height: 12),
              Expanded(child: _FactCard(card: card)),
              const SizedBox(height: 12),
              Text(
                'Mỗi lần quét – một kiến thức mới',
                style: textTheme.bodySmall?.copyWith(color: Colors.black45),
              ),
              const SizedBox(height: 12),
              OutlinedButton.icon(
                onPressed: () => Navigator.of(context).pop(),
                icon: const Icon(Icons.photo_camera_outlined, color: chemGreen),
                label: const Text('Quét vật khác'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _FactCard extends StatelessWidget {
  const _FactCard({required this.card});

  final Flashcard card;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: chemPurple.withValues(alpha: 0.25)),
        boxShadow: [
          BoxShadow(
            color: chemPurple.withValues(alpha: 0.08),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.science_outlined, color: chemPurple),
            const SizedBox(height: 12),
            Text(card.title,
                style: textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 12),
            Text(card.content, style: textTheme.bodyLarge?.copyWith(height: 1.5)),
            const SizedBox(height: 20),
            Text(
              'Nguồn: ${card.sourceTitle}',
              style: textTheme.bodySmall?.copyWith(color: Colors.black45),
            ),
          ],
        ),
      ),
    );
  }
}
