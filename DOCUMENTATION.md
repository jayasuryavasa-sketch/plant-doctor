# Project documentation

## Abstract

Plant Doctor is an AI-assisted web system that accepts a leaf photo and presents a possible plant condition with accessible, responsible care guidance. It combines a Flask backend, browser interface, SQLite history, a modular prediction layer, and a separate disease-information data layer. The core remains usable without model weights through an explicitly uncertain demo mode.

## Problem statement and solution

Home gardeners and students may struggle to recognize visible plant-health problems early. Plant Doctor provides a simple image upload/camera flow and organizes possible symptoms, causes, prevention, and next steps. It supports—not replaces—local agricultural expertise.

## Objectives

- Make image-based plant checks approachable.
- Clearly communicate uncertainty and limitations.
- Keep model, guidance data, UI, and storage modular.
- Offer a CPU-friendly transfer-learning path.
- Preserve a clean, responsive, accessible user experience.

## Modules

| Module | Responsibility |
|---|---|
| Frontend | Home, Scan, Result, Guide, History, About, camera and previews |
| Validation | Safe filename, allowed type, size, readable content, dimensions, darkness |
| Prediction | Demo fallback or trained MobileNetV3-small inference |
| Data layer | JSON maps model labels to human-readable guidance |
| Database | SQLite stores scan metadata and local image filename |

## System requirements

Windows 10/11, Python 3.10–3.12, 4 GB RAM minimum for the web app; more memory is helpful for CPU model training. Use VS Code and a current Chrome/Edge browser for camera testing.

## Final demonstration flow

1. Open the home page and explain the purpose and disclaimer.
2. Go to Scan Plant; show upload validation or camera capture.
3. Upload a clear leaf image and show the analysis state.
4. Explain the demo-mode uncertainty honestly, or trained-model confidence if a checkpoint exists.
5. Review symptoms, safe guidance, prevention, and limitations.
6. Show the Disease Guide and History.
7. Explain MobileNetV3 transfer learning, dataset split, testing, and future scope.

## Viva answers

**Why Python and Flask?** Python has strong image/ML libraries. Flask is small, readable, and practical for serving pages, uploads, predictions, and a local database.

**What is a CNN and transfer learning?** A CNN learns visual patterns from pixels. Transfer learning starts from a model already trained on broad images, then adapts its final layers to plant classes, reducing training cost.

**Why resize and normalize?** The model expects consistently sized, similarly scaled inputs. Augmentation creates modest visual variation during training to reduce overfitting.

**What are precision, recall, F1, and a confusion matrix?** Precision asks whether predicted cases were correct; recall asks whether real cases were found; F1 balances both; a confusion matrix shows which classes are confused.

**Is confidence certainty?** No. It is the model's relative score among its known classes, not a field diagnosis or guaranteed probability of truth.

**Why SQLite?** It is a lightweight local database that needs no separate server and suits a beginner demonstration.

**Limitations?** Training data may not represent real lighting, cultivars, plant stages, pests, multiple conditions, or unsupported species. Poor images and domain shift can produce false predictions.

## Future scope

Add multilingual/Telugu content, accounts, cloud database/storage, expert review, weather-aware reminders, explainable heatmaps, calibrated confidence, severity models with validated labels, and a mobile app. Each feature needs fresh security, privacy, and evaluation work.

## Conclusion

Plant Doctor demonstrates a stable, responsible foundation for AI-assisted plant-health education: clear inputs, careful validation, modular ML integration, actionable general guidance, and transparent limitations.
