import 'dart:math';

import 'package:chemsort/models/flashcard.dart';
import 'package:chemsort/models/waste_material.dart';
import 'package:chemsort/services/fact_service.dart';
import 'package:flutter_test/flutter_test.dart';

Flashcard _card(String id, String material,
        {int? code, String status = 'reviewed'}) =>
    Flashcard(
      id: id,
      materialKey: material,
      plasticCode: code,
      title: id,
      content: '',
      sourceTitle: '',
      sourceUrl: '',
      reviewStatus: status,
    );

void main() {
  final cards = [
    for (var i = 0; i < 4; i++) _card('glass_$i', 'glass'),
    _card('glass_draft', 'glass', status: 'draft'),
    _card('plastic_general', 'plastic'),
    _card('plastic_5_a', 'plastic', code: 5),
    _card('plastic_5_b', 'plastic', code: 5),
    _card('plastic_1_a', 'plastic', code: 1),
  ];

  test('never repeats the previous card while alternatives exist', () {
    for (var seed = 0; seed < 50; seed++) {
      final deck = FactService(cards, random: Random(seed))
          .deck(WasteMaterial.glass);
      var previous = deck.next();
      for (var i = 0; i < 40; i++) {
        final current = deck.next();
        expect(current.id, isNot(previous.id));
        previous = current;
      }
    }
  });

  test('shows every card once per round', () {
    final deck = FactService(cards, random: Random(1)).deck(WasteMaterial.glass);
    final round = {for (var i = 0; i < deck.size; i++) deck.next().id};
    expect(round, {'glass_0', 'glass_1', 'glass_2', 'glass_3'});
  });

  test('ignores cards that are not reviewed', () {
    final pool = FactService(cards).pool(WasteMaterial.glass);
    expect(pool.map((c) => c.id), isNot(contains('glass_draft')));
  });

  test('plastic code selects the resin pool; null selects general facts', () {
    final service = FactService(cards);
    expect(
      service.pool(WasteMaterial.plastic, plasticCode: 5).map((c) => c.id),
      unorderedEquals(['plastic_5_a', 'plastic_5_b']),
    );
    expect(
      service.pool(WasteMaterial.plastic).map((c) => c.id),
      ['plastic_general'],
    );
  });

  test('consecutive scans of one material never repeat a fact', () {
    for (var seed = 0; seed < 50; seed++) {
      final service = FactService(cards, random: Random(seed));
      var previous = service.nextFact(WasteMaterial.glass);
      for (var scan = 0; scan < 40; scan++) {
        // Scans of other materials in between must not reset the glass deck.
        service.nextFact(WasteMaterial.plastic, plasticCode: 5);
        final current = service.nextFact(WasteMaterial.glass);
        expect(current.id, isNot(previous.id));
        previous = current;
      }
    }
  });

  test('every fact of a pool is seen before any repeats', () {
    final service = FactService(cards, random: Random(3));
    final seen = {
      for (var i = 0; i < 4; i++) service.nextFact(WasteMaterial.glass).id,
    };
    expect(seen, hasLength(4));
  });

  test('empty pool fails loudly', () {
    expect(() => FactService(cards).deck(WasteMaterial.metal), throwsStateError);
  });
}
