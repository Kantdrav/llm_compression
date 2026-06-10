import 'package:flutter_test/flutter_test.dart';

import 'package:model_compressor_flutter/main.dart';

void main() {
  testWidgets('renders the compression dashboard', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());

    expect(find.text('Model Compressor'), findsWidgets);
    expect(find.text('Compress model'), findsOneWidget);
    expect(find.text('Choose model ZIP'), findsOneWidget);
  });
}
