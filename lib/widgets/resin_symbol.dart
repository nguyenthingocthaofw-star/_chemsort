import 'package:flutter/material.dart';

/// Recycling-triangle symbol with the resin code in the middle.
class ResinSymbol extends StatelessWidget {
  const ResinSymbol({super.key, required this.code, this.size = 44});

  final int code;
  final double size;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: size,
      height: size,
      child: Stack(
        alignment: Alignment.center,
        children: [
          Icon(Icons.change_history, size: size),
          Padding(
            padding: EdgeInsets.only(top: size * 0.22),
            child: Text(
              '$code',
              style: TextStyle(
                fontSize: size * 0.32,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
