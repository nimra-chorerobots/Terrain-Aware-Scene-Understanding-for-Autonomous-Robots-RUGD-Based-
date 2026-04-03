# Terrain-Aware Scene Understanding for Autonomous Robots (RUGD-Based)

## Overview
This project implements a semantic scene understanding system for autonomous robots using the RUGD dataset. The system analyzes outdoor environments and classifies terrain into drivable, obstacle, and hazard regions, enabling adaptive navigation decisions such as speed adjustment and risk assessment.

## Motivation
Autonomous robots operating in unstructured environments (e.g., parks, trails, construction sites) must understand terrain conditions to ensure safe navigation. This project demonstrates how semantic segmentation can be used for real-time decision-making.

## Key Features
- Semantic segmentation using RUGD annotations
- Terrain classification (grass, dirt, gravel, water, etc.)
- Scene understanding:
  - Drivable
  - Semi-drivable
  - Non-drivable
  - Hazard detection (e.g., water)
- Adaptive speed recommendation
- Visual overlay for perception debugging

## Dataset
RUGD (Rensselaer Unstructured Grid Dataset)

The dataset contains:
- RGB images
- Pixel-wise semantic annotations
- 24 terrain/object classes

## Methodology
1. Load RGB image and corresponding annotation
2. Decode annotation using RGB colormap
3. Map classes into:
   - Drivable terrain (grass, dirt, gravel, asphalt)
   - Obstacles (trees, rocks, bushes, structures)
   - Hazards (water)
4. Compute terrain distribution ratios
5. Generate scene classification
6. Output navigation decisions

## Output Example
- Green → Drivable terrain
- Red → Obstacles
- Blue → Hazard (water)
  
<img width="2048" height="853" alt="35504162-9583-4549-8941-15392f70a921" src="https://github.com/user-attachments/assets/26ca41a5-873c-4b0d-b05b-e86c952a3f04" />

## Installation

```bash
pip install -r requirements.txt
