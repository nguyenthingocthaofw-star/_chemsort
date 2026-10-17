import 'package:flutter/material.dart';

import '../models/resin.dart';
import '../models/waste_material.dart';
import '../theme.dart';
import '../widgets/resin_symbol.dart';
import 'flashcard_screen.dart';
import 'polymer_screen.dart';

/// Plastic branch: the user reads the resin code printed on the product.
/// The model never guesses the polymer from appearance.
class ResinScreen extends StatelessWidget {
  const ResinScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Chọn mã nhựa')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              const Text(
                'Tìm ký hiệu này trên sản phẩm',
                style: TextStyle(color: Colors.black54),
              ),
              const SizedBox(height: 16),
              Expanded(
                child: GridView.count(
                  crossAxisCount: 3,
                  mainAxisSpacing: 12,
                  crossAxisSpacing: 12,
                  children: [
                    for (final resin in Resin.all)
                      _ResinTile(
                        resin: resin,
                        onTap: () => Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => PolymerScreen(resin: resin),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              TextButton(
                style: TextButton.styleFrom(foregroundColor: chemGreen),
                onPressed: () =>
                    showFactForScan(context, WasteMaterial.plastic),
                child: const Text('Không thấy mã? Xem fact nhựa chung'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ResinTile extends StatelessWidget {
  const _ResinTile({required this.resin, required this.onTap});

  final Resin resin;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: BorderSide(color: Colors.grey.shade300),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ResinSymbol(code: resin.code),
            const SizedBox(height: 6),
            Text(resin.abbreviation,
                style: const TextStyle(fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}
