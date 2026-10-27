import 'dart:convert';
import 'dart:math';

import '../models/flashcard.dart';
import '../models/waste_material.dart';

class FactService {
  FactService(Iterable<Flashcard> cards, {Random? random})
      : _cards = cards.where((card) => card.isReviewed).toList(growable: false),
        _random = random ?? Random();

  factory FactService.fromJson(String json, {Random? random}) => FactService(
        (jsonDecode(json) as List)
            .map((item) => Flashcard.fromJson(item as Map<String, dynamic>)),
        random: random,
      );

  final List<Flashcard> _cards;
  final Random _random;

  /// One deck per pool, kept for the whole session so consecutive scans of
  /// the same material never show the same fact twice in a row.
  final Map<(String, int?), FactDeck> _decks = {};

  /// The single fact shown for one scan.
  Flashcard nextFact(WasteMaterial material, {int? plasticCode}) =>
      _decks.putIfAbsent(
        (material.key, plasticCode),
        () => deck(material, plasticCode: plasticCode),
      ).next();

  /// Pool for a material; for plastic, [plasticCode] 1–7 selects the resin
  /// pool and null selects the general plastic facts.
  List<Flashcard> pool(WasteMaterial material, {int? plasticCode}) => _cards
      .where((card) =>
          card.materialKey == material.key && card.plasticCode == plasticCode)
      .toList(growable: false);

  FactDeck deck(WasteMaterial material, {int? plasticCode}) {
    final cards = pool(material, plasticCode: plasticCode);
    if (cards.isEmpty) {
      throw StateError(
        'No reviewed flashcards for ${material.key} code=$plasticCode',
      );
    }
    return FactDeck(cards, _random);
  }
}

/// Shuffle-bag over one pool: every card appears once per round and the
/// same card is never shown twice in a row while the pool has alternatives.
class FactDeck {
  FactDeck(List<Flashcard> cards, this._random)
      : _cards = List.unmodifiable(cards);

  final List<Flashcard> _cards;
  final Random _random;
  final List<Flashcard> _bag = [];
  Flashcard? _last;

  int get size => _cards.length;

  Flashcard next() {
    if (_bag.isEmpty) {
      _bag
        ..addAll(_cards)
        ..shuffle(_random);
      if (_bag.length > 1 && identical(_bag.last, _last)) {
        _bag.insert(0, _bag.removeLast());
      }
    }
    return _last = _bag.removeLast();
  }
}
