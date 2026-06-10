# CI/CD Guide

This project uses GitHub Actions for continuous integration and delivery.

## Workflows

- `CI`: runs on pushes and pull requests to `main`.
  - Installs the Python package.
  - Runs `pytest`.
  - Runs the demo export command.
  - Runs Flutter analyze and widget tests.
  - Builds Flutter web.
  - Builds an Android debug APK.

- `Deploy Flutter Web`: runs on pushes to `main`.
  - Builds `flutter_app` for web.
  - Deploys `flutter_app/build/web` to GitHub Pages.

- `Release`: runs when you push a tag like `v1.0.0`.
  - Builds the Flutter web release zip.
  - Builds the Android release APK.
  - Publishes both files to a GitHub Release.

## First-Time GitHub Setup

1. Push this repository to GitHub.
2. In GitHub, open `Settings -> Pages`.
3. Set the Pages source to `GitHub Actions`.
4. Push to `main` and wait for the `Deploy Flutter Web` workflow.

## Creating A Release

Use a version tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The release workflow will create downloadable app artifacts.

## Local Commands

Python tests:

```bash
.venv/bin/pytest
```

Flutter checks:

```bash
cd flutter_app
flutter analyze
flutter test
flutter build web
flutter build apk --debug
```
