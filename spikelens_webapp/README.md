# SpikeLens Web App

Interactive static web visualization for precomputed SNN inference traces.

## Local setup

```bash
cd spikelens_webapp
npm install
npm run dev
```

Then open the printed local URL.

## Build

```bash
npm run build
npm run preview
```

## Deploy to GitHub Pages

Option 1: manual gh-pages branch:

```bash
npm run deploy
```

Option 2: use GitHub Pages Actions / Settings and publish the generated static site.

## Data

The app loads:

```text
public/data/manifest.json
public/data/traces/<model_id>/<sample>.json
```
