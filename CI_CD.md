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

- `Deploy Backend to Render`: runs on pushes to `main`.
  - Triggers a Render deploy hook for the FastAPI backend.

- `Deploy Flutter Web`: runs on pushes to `main`.
  - Builds `flutter_app` for web with `API_BASE_URL` injected at build time.
  - Deploys `flutter_app/build/web` to Vercel.

- `Release`: runs when you push a tag like `v1.0.0`.
  - Builds the Flutter web release zip.
  - Builds the Android release APK.
  - Publishes both files to a GitHub Release.

## First-Time GitHub Setup

1. Push this repository to GitHub.
2. Create a Render service for the backend and copy its deploy hook URL into `RENDER_DEPLOY_HOOK_URL`.
3. Create a Vercel project for the Flutter web app and store its token in `VERCEL_TOKEN`.
4. Set `API_BASE_URL` to the deployed backend URL so the web app calls the live API.
5. Push to `main` and wait for the deploy workflows.

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
