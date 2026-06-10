import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:path/path.dart' as path;

Future<Uint8List> buildZipFromDirectory(String directoryPath) async {
  final directory = Directory(directoryPath);
  if (!await directory.exists()) {
    throw FileSystemException('Directory does not exist', directoryPath);
  }

  final archive = Archive();
  final rootPath = path.normalize(directory.absolute.path);

  await for (final entity in directory.list(recursive: true, followLinks: false)) {
    if (entity is! File) {
      continue;
    }

    final relativePath = path.relative(entity.path, from: rootPath);
    final bytes = await entity.readAsBytes();
    archive.addFile(ArchiveFile(relativePath, bytes.length, bytes));
  }

  return Uint8List.fromList(ZipEncoder().encode(archive));
}
