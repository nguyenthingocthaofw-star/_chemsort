/// Vietnamese household solid-waste groups (Luật BVMT 2020, Điều 75).
enum VnGroup {
  group1('Nhóm I', 'Chất thải có khả năng tái sử dụng, tái chế'),
  group2('Nhóm II', 'Chất thải thực phẩm');

  const VnGroup(this.label, this.description);

  final String label;
  final String description;
}

/// The six app-facing classes, in the order of `app_output_6class`.
///
/// [appClass] must match `chemsort_mapping.json`; [key] matches
/// `material_key` in `flashcards.json`.
enum WasteMaterial {
  paperCardboard(
    key: 'paper_cardboard',
    appClass: 'Paper/Cardboard',
    titleEn: 'PAPER / CARDBOARD',
    titleVi: 'Giấy / Bìa carton',
    vnGroup: VnGroup.group1,
    hint: 'Giấy chủ yếu được tạo từ cellulose, một polymer tự nhiên.',
  ),
  organic(
    key: 'organic',
    appClass: 'Organic',
    titleEn: 'ORGANIC',
    titleVi: 'Thực phẩm',
    vnGroup: VnGroup.group2,
    hint: 'Thực phẩm chứa carbohydrate, protein và lipid.',
  ),
  glass(
    key: 'glass',
    appClass: 'Glass',
    titleEn: 'GLASS',
    titleVi: 'Thủy tinh',
    vnGroup: VnGroup.group1,
    hint: 'Thủy tinh thông dụng được tạo chủ yếu từ silica (SiO₂).',
  ),
  metal(
    key: 'metal',
    appClass: 'Metal',
    titleEn: 'METAL',
    titleVi: 'Kim loại',
    vnGroup: VnGroup.group1,
    hint: 'Kim loại có thể bị oxi hóa, tạo lớp oxide trên bề mặt.',
  ),
  other(
    key: 'other',
    appClass: 'Other',
    titleEn: 'OTHER',
    titleVi: 'Chưa xác định',
    vnGroup: null,
    hint: 'Vật thể nằm ngoài 5 nhóm vật liệu chính mà model nhận diện.',
  ),
  plastic(
    key: 'plastic',
    appClass: 'Plastic',
    titleEn: 'PLASTIC',
    titleVi: 'Nhựa',
    vnGroup: VnGroup.group1,
    hint: 'Nhựa gồm nhiều loại polymer. Hãy tìm mã nhựa trên sản phẩm.',
  );

  const WasteMaterial({
    required this.key,
    required this.appClass,
    required this.titleEn,
    required this.titleVi,
    required this.vnGroup,
    required this.hint,
  });

  final String key;
  final String appClass;
  final String titleEn;
  final String titleVi;

  /// Null means the app does not have enough information to assign a group;
  /// "Other" is deliberately not mapped to Nhóm III.
  final VnGroup? vnGroup;
  final String hint;

  static WasteMaterial fromKey(String key) => values.firstWhere(
        (material) => material.key == key,
        orElse: () => throw FormatException('Unknown material key: $key'),
      );

  static WasteMaterial fromAppClass(String appClass) =>
      values.firstWhere(
        (material) => material.appClass == appClass,
        orElse: () => throw FormatException('Unknown app class: $appClass'),
      );
}
