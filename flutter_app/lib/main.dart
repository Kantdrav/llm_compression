import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'archive_builder.dart';

const String kApiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://127.0.0.1:8000',
);

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Model Compressor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF18A999),
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF07111D),
        useMaterial3: true,
      ),
      home: const CompressionHomePage(),
    );
  }
}

class CompressionHomePage extends StatefulWidget {
  const CompressionHomePage({super.key});

  @override
  State<CompressionHomePage> createState() => _CompressionHomePageState();
}

class _CompressionHomePageState extends State<CompressionHomePage> {
  final TextEditingController _backendUrlController = TextEditingController(text: kApiBaseUrl);
  final TextEditingController _modelIdController = TextEditingController(text: 'local-model');
  final TextEditingController _remoteModelIdController = TextEditingController(text: 'gpt2');
  final TextEditingController _remoteRevisionController = TextEditingController();

  String? _selectedArchiveName;
  List<int>? _selectedArchiveBytes;
  String? _selectedDirectory;
  bool _isUploading = false;
  bool _useRemoteModel = false;
  bool _trustRemoteCode = false;
  String? _statusMessage;
  String? _errorMessage;
  Map<String, dynamic>? _result;
  String _method = 'tensor_inspired';
  double _rankRatio = 0.5;
  double _healSteps = 0;
  String _remotePreset = 'gpt2';

  @override
  void dispose() {
    _backendUrlController.dispose();
    _modelIdController.dispose();
    _remoteModelIdController.dispose();
    _remoteRevisionController.dispose();
    super.dispose();
  }

  String _trimTrailingSlashes(String value) => value.trim().replaceAll(RegExp(r'/+$'), '');

  void _setRemotePreset(String value) {
    setState(() {
      _remotePreset = value;
      _remoteModelIdController.text = value;
    });
  }

  Future<void> _pickArchive() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const ['zip'],
        withData: true,
      );
      if (result == null || result.files.isEmpty) {
        return;
      }

      final file = result.files.single;
      final bytes = file.bytes;
      if (bytes == null) {
        setState(() {
          _errorMessage = 'Could not read the selected ZIP file. Try a different file picker path or platform.';
          _statusMessage = null;
        });
        return;
      }

      setState(() {
        _selectedArchiveName = file.name;
        _selectedArchiveBytes = bytes;
        _selectedDirectory = null;
        _errorMessage = null;
        _statusMessage = 'Selected ZIP: ${file.name}';
      });
    } catch (error) {
      setState(() {
        _errorMessage = 'ZIP selection failed: $error';
        _statusMessage = null;
      });
    }
  }

  Future<void> _pickDirectory() async {
    try {
      final directory = await FilePicker.platform.getDirectoryPath();
      if (directory == null || directory.isEmpty) {
        return;
      }

      setState(() {
        _selectedDirectory = directory;
        _selectedArchiveName = null;
        _selectedArchiveBytes = null;
        _errorMessage = null;
        _statusMessage = 'Selected folder: $directory';
      });
    } catch (error) {
      setState(() {
        _errorMessage = 'Directory selection failed: $error';
        _statusMessage = null;
      });
    }
  }

  Future<_UploadPayload> _buildPayload() async {
    final archiveBytes = _selectedArchiveBytes;
    final archiveName = _selectedArchiveName;
    if (archiveBytes != null && archiveName != null) {
      return _UploadPayload(name: archiveName, bytes: archiveBytes);
    }

    final directory = _selectedDirectory;
    if (directory != null) {
      final bytes = await buildZipFromDirectory(directory);
      final directoryName = directory.split(RegExp(r'[\\/]')).where((segment) => segment.isNotEmpty).last;
      return _UploadPayload(name: '$directoryName.zip', bytes: bytes);
    }

    throw StateError('Choose a ZIP archive or a folder before compressing.');
  }

  Future<void> _compressModel() async {
    final backendUrl = _trimTrailingSlashes(_backendUrlController.text);
    if (backendUrl.isEmpty) {
      setState(() {
        _errorMessage = 'Enter a backend URL.';
        _statusMessage = null;
      });
      return;
    }

    if (_useRemoteModel) {
      await _compressRemoteModel(backendUrl);
      return;
    }

    if (_selectedArchiveName == null && _selectedDirectory == null) {
      setState(() {
        _errorMessage = 'Choose a zip archive or folder before compressing.';
        _statusMessage = null;
      });
      return;
    }

    setState(() {
      _isUploading = true;
      _statusMessage = 'Uploading and compressing...';
      _errorMessage = null;
      _result = null;
    });

    try {
      final payload = await _buildPayload();
      final request = http.MultipartRequest('POST', Uri.parse('$backendUrl/compress-upload'));
      request.fields.addAll({
        'model_id': _modelIdController.text.trim().isEmpty ? 'local-model' : _modelIdController.text.trim(),
        'method': _method,
        'rank_ratio': _rankRatio.toStringAsFixed(2),
        'target_device': 'cpu',
        'heal_steps': _healSteps.round().toString(),
        'trust_remote_code': _trustRemoteCode.toString(),
      });
      request.files.add(
        http.MultipartFile.fromBytes(
          'model_archive',
          payload.bytes,
          filename: payload.name,
        ),
      );

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (!mounted) {
        return;
      }

      if (response.statusCode < 200 || response.statusCode >= 300) {
        setState(() {
          _errorMessage = response.body.isEmpty ? 'Compression failed with HTTP ${response.statusCode}.' : response.body;
          _statusMessage = null;
        });
        return;
      }

      final decoded = jsonDecode(response.body) as Map<String, dynamic>;
      setState(() {
        _result = decoded;
        _statusMessage = 'Compression complete.';
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage = error.toString();
        _statusMessage = null;
      });
    } finally {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
      }
    }
  }

  Future<void> _compressRemoteModel(String backendUrl) async {
    final remoteModelId = _remoteModelIdController.text.trim();
    if (remoteModelId.isEmpty) {
      setState(() {
        _errorMessage = 'Enter a Hugging Face model ID before compressing.';
        _statusMessage = null;
      });
      return;
    }

    setState(() {
      _isUploading = true;
      _statusMessage = 'Downloading and compressing...';
      _errorMessage = null;
      _result = null;
    });

    try {
      final response = await http.post(
        Uri.parse('$backendUrl/compress-remote'),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode({
          'model_id': remoteModelId,
          'revision': _remoteRevisionController.text.trim().isEmpty ? null : _remoteRevisionController.text.trim(),
          'method': _method,
          'rank_ratio': _rankRatio,
          'target_device': 'cpu',
          'trust_remote_code': _trustRemoteCode,
          'heal_steps': _healSteps.round(),
        }),
      );

      if (!mounted) {
        return;
      }

      if (response.statusCode < 200 || response.statusCode >= 300) {
        setState(() {
          _errorMessage = response.body.isEmpty ? 'Remote compression failed with HTTP ${response.statusCode}.' : response.body;
          _statusMessage = null;
        });
        return;
      }

      final decoded = jsonDecode(response.body) as Map<String, dynamic>;
      setState(() {
        _result = decoded;
        _statusMessage = 'Remote compression complete.';
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _errorMessage = error.toString();
        _statusMessage = null;
      });
    } finally {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      body: DecoratedBox(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF07111D), Color(0xFF0B2A36), Color(0xFF153B3A)],
          ),
        ),
        child: Stack(
          children: [
            const Positioned(
              top: -120,
              left: -80,
              child: _GlowBlob(color: Color(0xFF18A999), size: 260),
            ),
            const Positioned(
              bottom: -100,
              right: -60,
              child: _GlowBlob(color: Color(0xFFE6A34A), size: 220),
            ),
            SafeArea(
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 1100),
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const SizedBox(height: 16),
                        Text(
                          'Model Compressor',
                          style: theme.textTheme.displaySmall?.copyWith(
                            fontWeight: FontWeight.w800,
                            letterSpacing: -1.2,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Upload a zipped Hugging Face model directory, choose a compression method, and send it to the FastAPI backend.',
                          style: theme.textTheme.titleMedium?.copyWith(color: Colors.white70),
                        ),
                        const SizedBox(height: 28),
                        LayoutBuilder(
                          builder: (context, constraints) {
                            final isWide = constraints.maxWidth >= 900;
                            final leftColumn = _buildControlsCard(context);
                            final rightColumn = _buildResultCard(context);
                            if (isWide) {
                              return Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Expanded(child: leftColumn),
                                  const SizedBox(width: 20),
                                  Expanded(child: rightColumn),
                                ],
                              );
                            }
                            return Column(
                              children: [
                                leftColumn,
                                const SizedBox(height: 20),
                                rightColumn,
                              ],
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildControlsCard(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      color: const Color(0xFF0B1828).withOpacity(0.92),
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Compress a model', style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 20),
            TextField(
              controller: _backendUrlController,
              decoration: const InputDecoration(
                labelText: 'Backend URL',
                hintText: 'http://127.0.0.1:8000',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            ToggleButtons(
              isSelected: [_useRemoteModel == false, _useRemoteModel],
              onPressed: (index) {
                setState(() {
                  _useRemoteModel = index == 1;
                });
              },
              borderRadius: BorderRadius.circular(14),
              constraints: const BoxConstraints(minHeight: 44),
              children: const [
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16),
                  child: Text('Local upload'),
                ),
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16),
                  child: Text('Internet model'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _modelIdController,
              decoration: const InputDecoration(
                labelText: 'Bundle label',
                hintText: 'llama-demo or custom-upload',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              value: _method,
              items: const [
                DropdownMenuItem(value: 'tensor_inspired', child: Text('Tensor-inspired')),
                DropdownMenuItem(value: 'quantize', child: Text('Dynamic quantization')),
                DropdownMenuItem(value: 'mpo', child: Text('MPO')),
              ],
              onChanged: (value) {
                if (value == null) {
                  return;
                }
                setState(() {
                  _method = value;
                });
              },
              decoration: const InputDecoration(
                labelText: 'Compression method',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            CheckboxListTile(
              contentPadding: EdgeInsets.zero,
              controlAffinity: ListTileControlAffinity.leading,
              value: _trustRemoteCode,
              onChanged: _isUploading ? null : (value) => setState(() => _trustRemoteCode = value ?? false),
              title: const Text('Trust remote code'),
              subtitle: const Text('Enable this for Hugging Face models that ship custom loading code.'),
            ),
            const SizedBox(height: 18),
            Text('Rank ratio: ${_rankRatio.toStringAsFixed(2)}', style: theme.textTheme.titleMedium),
            Slider(
              value: _rankRatio,
              min: 0.1,
              max: 1.0,
              divisions: 9,
              onChanged: (value) => setState(() => _rankRatio = value),
            ),
            const SizedBox(height: 8),
            Text('Healing steps: ${_healSteps.round()}', style: theme.textTheme.titleMedium),
            Slider(
              value: _healSteps,
              min: 0,
              max: 8,
              divisions: 8,
              onChanged: (value) => setState(() => _healSteps = value),
            ),
            const SizedBox(height: 12),
            if (_useRemoteModel) ...[
              DropdownButtonFormField<String>(
                value: _remotePreset,
                items: const [
                  DropdownMenuItem(value: 'gpt2', child: Text('gpt2')),
                  DropdownMenuItem(value: 'distilgpt2', child: Text('distilgpt2')),
                  DropdownMenuItem(value: 'sshleifer/tiny-gpt2', child: Text('sshleifer/tiny-gpt2')),
                  DropdownMenuItem(value: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', child: Text('TinyLlama/TinyLlama-1.1B-Chat-v1.0')),
                ],
                onChanged: _isUploading
                    ? null
                    : (value) {
                        if (value != null) {
                          _setRemotePreset(value);
                        }
                      },
                decoration: const InputDecoration(
                  labelText: 'Popular model IDs',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _remoteModelIdController,
                decoration: const InputDecoration(
                  labelText: 'Hugging Face model ID',
                  hintText: 'meta-llama/Llama-3.2-1B-Instruct',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _remoteRevisionController,
                decoration: const InputDecoration(
                  labelText: 'Revision or tag',
                  hintText: 'main, v1.0, or a commit hash',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              const Text(
                'The backend will download the model from Hugging Face, compress it, and return a manifest path for the bundle.',
                style: TextStyle(color: Colors.white70),
              ),
            ] else ...[
              OutlinedButton.icon(
                onPressed: _isUploading ? null : _pickArchive,
                icon: const Icon(Icons.upload_file),
                label: const Text('Choose model ZIP'),
              ),
              const SizedBox(height: 10),
              OutlinedButton.icon(
                onPressed: _isUploading ? null : _pickDirectory,
                icon: const Icon(Icons.folder_open),
                label: const Text('Choose model folder'),
              ),
              const SizedBox(height: 12),
              if (_selectedArchiveName != null) _InfoChip(label: 'Selected file', value: _selectedArchiveName!),
              if (_selectedDirectory != null)
                const SizedBox(height: 12),
              if (_selectedDirectory != null)
                _InfoChip(label: 'Selected folder', value: _selectedDirectory!),
            ],
            const SizedBox(height: 20),
            FilledButton.icon(
              onPressed: _isUploading ? null : _compressModel,
              icon: _isUploading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.precision_manufacturing),
              label: Text(_isUploading ? 'Compressing...' : 'Compress model'),
              style: FilledButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
            ),
            const SizedBox(height: 16),
            if (_statusMessage != null)
              Text(_statusMessage!, style: theme.textTheme.bodyMedium?.copyWith(color: Colors.greenAccent)),
            if (_errorMessage != null)
              Text(_errorMessage!, style: theme.textTheme.bodyMedium?.copyWith(color: Colors.redAccent)),
            const SizedBox(height: 8),
            const Text(
              'Local mode expects a ZIP archive or folder containing a Hugging Face model directory with config.json and weights.',
              style: TextStyle(color: Colors.white70),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard(BuildContext context) {
    final theme = Theme.of(context);
    final result = _result;
    final metrics = result?['metrics'] as Map<String, dynamic>?;
    return Card(
      color: const Color(0xFF0B1828).withOpacity(0.92),
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('Compression result', style: theme.textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 16),
            if (result == null)
              const Text(
                'Run a compression job to see the manifest path, output bundle, and metrics here.',
                style: TextStyle(color: Colors.white70),
              )
            else ...[
              _InfoChip(label: 'Manifest', value: result['manifest']?.toString() ?? '-'),
              const SizedBox(height: 12),
              _InfoChip(label: 'Output dir', value: result['output_dir']?.toString() ?? '-'),
              const SizedBox(height: 12),
              _InfoChip(label: 'Parameter ratio', value: result['parameter_ratio']?.toString() ?? '-'),
              const SizedBox(height: 20),
              Text('Metrics', style: theme.textTheme.titleMedium),
              const SizedBox(height: 10),
              DecoratedBox(
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(18),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: SelectableText(
                    const JsonEncoder.withIndent('  ').convert(metrics ?? const {}),
                    style: const TextStyle(fontFamily: 'monospace', color: Colors.white70),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _UploadPayload {
  const _UploadPayload({required this.name, required this.bytes});

  final String name;
  final List<int> bytes;
}

class _GlowBlob extends StatelessWidget {
  const _GlowBlob({required this.color, required this.size});

  final Color color;
  final double size;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: RadialGradient(
          colors: [color.withOpacity(0.5), color.withOpacity(0.05)],
        ),
      ),
    );
  }
}

class _InfoChip extends StatelessWidget {
  const _InfoChip({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.06),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(color: Colors.white60, fontSize: 12)),
          const SizedBox(height: 4),
          SelectableText(
            value,
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}
