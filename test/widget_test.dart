import 'package:chemsort/models/flashcard.dart';
import 'package:chemsort/screens/flashcard_screen.dart';
import 'package:chemsort/screens/home_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('home screen shows the start button', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: HomeScreen()));

    expect(find.text('ChemSort'), findsOneWidget);
    expect(find.text('Bắt đầu'), findsOneWidget);
  });

  testWidgets('flashcard offers a new scan, not another fact', (tester) async {
    const card = Flashcard(
      id: 'glass_001',
      materialKey: 'glass',
      plasticCode: null,
      title: 'Thủy tinh và silica',
      content: 'Nội dung',
      sourceTitle: 'Nguồn',
      sourceUrl: 'https://example.org',
      reviewStatus: 'reviewed',
    );
    await tester.pumpWidget(const MaterialApp(
      home: FlashcardScreen(card: card, topic: 'Thủy tinh'),
    ));

    expect(find.text('Thủy tinh và silica'), findsOneWidget);
    expect(find.text('Quét vật khác'), findsOneWidget);
    expect(find.text('Fact khác'), findsNothing);
  });
}
