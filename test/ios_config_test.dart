import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  group('iOS project configuration', () {
    test('declares the permissions used by the image picker', () {
      final infoPlist = File('ios/Runner/Info.plist').readAsStringSync();

      expect(infoPlist, contains('<key>NSCameraUsageDescription</key>'));
      expect(infoPlist, contains('<key>NSPhotoLibraryUsageDescription</key>'));
    });

    test('uses CocoaPods with the supported deployment target', () {
      final podfile = File('ios/Podfile').readAsStringSync();
      final pubspec = File('pubspec.yaml').readAsStringSync();

      expect(podfile, contains("platform :ios, '15.0'"));
      expect(podfile, contains('flutter_install_all_ios_pods'));
      expect(pubspec, contains('enable-swift-package-manager: false'));
    });

    test('preserves TFLite symbols in release archives', () {
      final project =
          File('ios/Runner.xcodeproj/project.pbxproj').readAsStringSync();

      expect(
        'STRIP_STYLE = "non-global";'.allMatches(project),
        hasLength(2),
      );
    });
  });
}
