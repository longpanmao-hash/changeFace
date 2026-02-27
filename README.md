# Face Swap Project

This repository now contains two routes:

1. `TypeScript` lightweight swap (`src/face-swap.ts`)
2. `InsightFace + InSwapper` deep swap (recommended)

## Deep Swap (PNG only)

Deep swap requires Python.

### 1. Install Python

Install Python `3.10/3.11/3.12` and ensure `python` is available in terminal.

### 2. Setup deep model env

```bash
npm run deep:setup
```

This command will:
- install `deep/requirements.txt`
- download `inswapper_128.onnx` to `deep/models/`

If setup says C++ Build Tools are missing, install once:

```bash
winget install Microsoft.VisualStudio.2022.BuildTools
```

Deep setup now also installs `mediapipe` for cleaner head/person segmentation in `headpaste` mode.

### 3. Run deep face swap

```bash
npm run deep:swap -- -Source ./images/source.png -Target ./images/target.png -Output ./output/result.png
```

Notes:
- input/output must be `.png`
- script swaps the largest face from source to largest face in target

Fullface mode (stronger, includes face-shape region):

```bash
npm run deep:swap -- -Source ./images/source.png -Target ./images/target.png -Output ./output/result_fullface.png -Mode fullface
```

Headpaste mode (closest to "move source head over target"):

```bash
npm run deep:swap -- -Source ./images/source.png -Target ./images/target.png -Output ./output/result_headpaste.png -Mode headpaste
```

Make headpaste more natural (lower strength):

```bash
npm run deep:swap -- -Source ./images/source.png -Target ./images/target.png -Output ./output/result_headpaste_natural.png -Mode headpaste -SourceRefs "" 0.45
```

Improve identity similarity with multiple source refs (same person, PNG):

```bash
npm run deep:swap -- -Source ./images/source.png -Target ./images/target.png -Output ./output/result_like.png -Mode fullface -SourceRefs "./images/ref1.png,./images/ref2.png,./images/ref3.png"
```

## Existing TypeScript Route

```bash
npm run swap -- --source ./images/source.jpg --target ./images/target.jpg --output ./output/result.jpg 0.75
```

The last positional value is `strength` (`0~1`).
