# GitHub Pages Deployment

The GitHub Pages deployment hosts the Flutter web interface. Compression,
model downloads, and inference continue to run on the FastAPI backend because
GitHub Pages cannot execute Python or PyTorch.

## Required setup

1. Deploy the backend using the existing Render workflow or another
   Python-capable host.
2. Configure the backend `CORS_ORIGINS` environment variable to include the
   GitHub Pages origin, such as `https://kantdrav.github.io`.
3. In the repository settings, open **Pages** and select **GitHub Actions** as
   the source.
4. Add the repository secret `API_BASE_URL` containing the public backend URL,
   for example `https://your-backend.example.com`.

Every push to `main` runs
`.github/workflows/deploy-github-pages.yml`. It builds Flutter with the
repository path as its base URL and publishes the result to:

```text
https://kantdrav.github.io/llm_compression/
```

The repository owner or name changes this URL. GitHub Actions exposes the
actual URL in the Pages deployment environment.

## Local production build

From the repository root:

```bash
cd flutter_app
flutter pub get
flutter build web --release \
  --base-href /llm_compression/ \
  --dart-define=API_BASE_URL=https://your-backend.example.com
```

The generated site is in `flutter_app/build/web`. The same frontend can be
deployed to Vercel using the existing Vercel workflow.
