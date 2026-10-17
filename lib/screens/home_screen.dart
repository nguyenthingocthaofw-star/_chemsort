import 'package:flutter/material.dart';

import '../theme.dart';
import 'scan_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
          child: Column(
            children: [
              Text(
                'ChemSort',
                style: textTheme.headlineMedium?.copyWith(
                  color: chemGreen,
                  fontWeight: FontWeight.w800,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Scan – Classify – Discover – Learn',
                style: textTheme.bodyMedium?.copyWith(color: Colors.black54),
              ),
              const Spacer(),
              Container(
                width: 160,
                height: 160,
                decoration: const BoxDecoration(
                  color: chemGreenLight,
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.science_outlined,
                    size: 84, color: chemGreen),
              ),
              const SizedBox(height: 32),
              Text(
                'Scan rác – Học Hóa',
                style: textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              Text(
                'Mỗi lần quét\nMột kiến thức mới',
                textAlign: TextAlign.center,
                style: textTheme.bodyLarge?.copyWith(color: Colors.black54),
              ),
              const Spacer(),
              FilledButton(
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute(
                    settings: const RouteSettings(name: ScanScreen.routeName),
                    builder: (_) => const ScanScreen(),
                  ),
                ),
                child: const Text('Bắt đầu'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
