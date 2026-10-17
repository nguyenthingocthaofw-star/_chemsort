/// One verified chemistry fact from `flashcards.json`.
class Flashcard {
  const Flashcard({
    required this.id,
    required this.materialKey,
    required this.plasticCode,
    required this.title,
    required this.content,
    required this.sourceTitle,
    required this.sourceUrl,
    required this.reviewStatus,
  });

  factory Flashcard.fromJson(Map<String, dynamic> json) => Flashcard(
        id: json['id'] as String,
        materialKey: json['material_key'] as String,
        plasticCode: json['plastic_code'] as int?,
        title: json['title_vi'] as String,
        content: json['content_vi'] as String,
        sourceTitle: json['source_title'] as String,
        sourceUrl: json['source_url'] as String,
        reviewStatus: json['review_status'] as String,
      );

  final String id;
  final String materialKey;

  /// Null for material-level facts, including general plastic facts.
  final int? plasticCode;
  final String title;
  final String content;
  final String sourceTitle;
  final String sourceUrl;
  final String reviewStatus;

  bool get isReviewed => reviewStatus == 'reviewed';
}
