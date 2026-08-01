# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository currently contains only planning docs ([Plan.md](Plan.md), [STEP.md](STEP.md)) — no source code, package manifest, build tooling, or dependencies exist yet. There are no build/lint/test commands to run because nothing has been scaffolded. When implementation begins, this file should be updated with the actual commands and architecture.

Implementation proceeds in small, incremental steps tracked in [STEP.md](STEP.md) — check that file for the current step and what's already working before starting new work.

## Chosen tech stack

- UI: React + TypeScript + Vite
- Inference: TensorFlow.js (runs pretrained models and reads intermediate-layer activations entirely in-browser)
- Hosting: GitHub Pages (static build only, no backend)

## What this project is

CNN Visualizer: an educational web app that visualizes how a convolutional neural network classifies images. It shows how the network progresses from simple to increasingly complex features across layers, and lets users intuitively see what part of an image the model focused on when making its decision.

Educational goals (from [Plan.md](Plan.md)):
- Teach the basic principles of CNN-based image classification
- Show how neuron activations behave at each layer
- Demonstrate the progression from simple to complex feature detection across layers
- Show how responses to the same image differ across different trained models

Planned user flow:
1. User selects one of several available models to classify with.
2. User captures or uploads an image to classify.
3. The app shows classification probabilities for what's in the image.
4. The user browses each layer's neuron activations (overlaid on the image), swiping between layers.

## Design constraints (binding for implementation decisions)

- **Client-side only, static hosting** — must run entirely in the browser with no backend, deployable to GitHub Pages.
- **TypeScript-based.**
- **Multiple switchable models** — architecture should not hardcode a single model.
- **Layer-by-layer activation overlay** — per-layer neuron response visualized as an overlay on the source image; users swipe between layers to preview them.
- **Education first**: when trading off implementation convenience vs. how clearly "how the AI is classifying this image" comes across, prioritize clarity.
- **UI should stay as simple as possible.**
- **MVP discipline**: do not add features beyond the MVP scope in Plan.md. Get the MVP working before adding anything else.
