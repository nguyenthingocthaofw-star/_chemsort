import 'package:flutter/material.dart';

import '../models/waste_material.dart';
import '../theme.dart';

/// Shows the Vietnamese waste group, or an explicit "not enough information"
/// message for Other (never Nhóm III).
class GroupBadge extends StatelessWidget {
  const GroupBadge({super.key, required this.material});

  final WasteMaterial material;

  @override
  Widget build(BuildContext context) {
    final group = material.vnGroup;
    final known = group != null;
    final color = known ? chemGreen : Colors.grey.shade700;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: known ? chemGreenLight : Colors.grey.shade100,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(known ? Icons.recycling : Icons.help_outline,
                  size: 18, color: color),
              const SizedBox(width: 6),
              Text(
                known
                    ? group.label +
                        (material == WasteMaterial.plastic
                            ? ' (phân loại chung)'
                            : '')
                    : 'Chưa đủ thông tin',
                style: TextStyle(color: color, fontWeight: FontWeight.w700),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            known
                ? group.description
                : 'Chưa đủ thông tin để xác định nhóm phân loại.',
            style: TextStyle(color: color),
          ),
        ],
      ),
    );
  }
}
