import 'package:flutter/material.dart';

import '../models/resin.dart';
import '../models/waste_material.dart';
import '../widgets/group_badge.dart';
import '../widgets/resin_symbol.dart';
import 'flashcard_screen.dart';

class PolymerScreen extends StatelessWidget {
  const PolymerScreen({super.key, required this.resin});

  final Resin resin;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: Text('Mã ${resin.code}')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Center(child: ResinSymbol(code: resin.code, size: 96)),
            const SizedBox(height: 12),
            Text(
              '${resin.code} – ${resin.abbreviation}',
              textAlign: TextAlign.center,
              style: textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800),
            ),
            Text(
              resin.name,
              textAlign: TextAlign.center,
              style: textTheme.titleMedium?.copyWith(color: Colors.black54),
            ),
            const SizedBox(height: 8),
            Text(resin.detail, textAlign: TextAlign.center),
            const SizedBox(height: 20),
            const GroupBadge(material: WasteMaterial.plastic),
            const SizedBox(height: 8),
            Text(
              'Mã nhựa chỉ cho biết loại resin/polymer, '
              'không quyết định nhóm phân loại rác.',
              style: textTheme.bodySmall?.copyWith(color: Colors.black45),
            ),
            const SizedBox(height: 24),
            FilledButton(
              onPressed: () => showFactForScan(
                context,
                WasteMaterial.plastic,
                plasticCode: resin.code,
              ),
              child: const Text('Xem fact'),
            ),
          ],
        ),
      ),
    );
  }
}
