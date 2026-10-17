/// Resin identification codes 1–7. The code only selects a fact pool; it is
/// never used to decide the Vietnamese waste group.
class Resin {
  const Resin(this.code, this.abbreviation, this.name, this.detail);

  final int code;
  final String abbreviation;
  final String name;
  final String detail;

  static const all = [
    Resin(1, 'PET', 'Polyethylene terephthalate', 'Polyester, có liên kết ester'),
    Resin(2, 'HDPE', 'High-density polyethylene', 'Polyethylene ít phân nhánh, mật độ cao'),
    Resin(3, 'PVC', 'Polyvinyl chloride', 'Polymer chứa nguyên tử Cl'),
    Resin(4, 'LDPE', 'Low-density polyethylene', 'Polyethylene nhiều phân nhánh'),
    Resin(5, 'PP', 'Polypropylene', 'Mạch carbon có nhóm –CH₃'),
    Resin(6, 'PS', 'Polystyrene', 'Mạch carbon có nhóm phenyl'),
    Resin(7, 'OTHER', 'Các loại nhựa khác', 'Resin không thuộc mã 1–6'),
  ];
}
