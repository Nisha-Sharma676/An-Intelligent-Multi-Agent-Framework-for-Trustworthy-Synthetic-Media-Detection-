# An Intelligent Multi-Agent Framework for Trustworthy Synthetic Media Detection 

## Overview

This project presents a multi-agent deepfake detection framework designed to detect manipulated audio-visual content by analyzing different modalities and combining their predictions through a fusion mechanism.

Instead of depending on a single detection model, the framework uses multiple specialized agents to examine different aspects of multimedia content. Their outputs are combined to produce a final prediction, with additional mechanisms for reliability assessment, uncertainty analysis, calibration and temporal localization.

The project also explores the behavior of individual agents, the contribution of different modalities, and the performance of alternative fusion strategies.

## Objectives

- Develop a multi-agent framework for audio-visual deepfake detection.
- Analyze different modalities independently and through combined predictions.
- Implement and evaluate fusion strategies for aggregating agent predictions.
- Investigate reliability-aware fusion to account for differences in agent reliability.
- Perform calibration and uncertainty analysis to study prediction confidence.
- Evaluate the contribution of individual agents and modalities through ablation experiments.
- Explore temporal localization of manipulated segments within videos.
- Compare baseline and proposed approaches through systematic evaluation.

## Datasets

The framework is developed and evaluated using the following datasets:

### FakeAVCeleb
A dataset containing real and manipulated audio-visual samples, used for deepfake detection and evaluation of the proposed framework.

### LAV-DF
A dataset designed for audio-visual deepfake detection and temporal localization, used to investigate manipulated content and identify suspicious temporal segments.

Dataset preparation, feature handling, evaluation splits and preprocessing are managed through the project pipeline.

> Datasets and raw video/audio files are not included in this repository. Users must obtain the datasets from their respective sources and prepare them according to the project pipeline.

## Proposed Methodology

### 1. Multi-Agent Detection

The framework uses specialized detection agents to analyze available audio-visual information. Each agent provides its own prediction, allowing different detection signals to be examined independently.

The individual agent outputs are further used for comparison, analysis and final prediction aggregation.

### 2. Agent-Level Analysis

Individual agent predictions are evaluated to understand their detection behavior, strengths, limitations and contribution to the overall framework.

Agent comparison and reliability analysis help investigate differences in performance across detection components.

### 3. Fusion Strategies

Multiple agent predictions are combined to generate a final detection decision.

The project investigates alternative fusion mechanisms, including Judge Fusion and reliability-aware fusion, to examine how individual predictions can be aggregated and how agent reliability can influence the final decision.

### 4. Reliability-Aware Fusion

Reliability-aware fusion explores the use of agent reliability information during prediction aggregation.

This approach investigates whether accounting for the quality and reliability of individual agents can improve the final decision compared with treating all agent outputs equally.

### 5. Calibration Analysis

Calibration experiments analyze whether the model's confidence is aligned with its actual predictive correctness.

Calibration audits and threshold analysis are used to study confidence behavior, prediction reliability and decision-making under different thresholds.

### 6. Uncertainty Analysis

The framework includes uncertainty analysis to investigate prediction confidence, uncertain samples and the relationship between confidence and model decisions.

Uncertainty-based evaluation helps analyze how the system behaves when predictions are less certain.

### 7. Modality Ablation

Modality ablation experiments examine the contribution of individual modalities and combinations of modalities to the final detection performance.

These experiments help identify the importance of different information sources within the multi-agent framework.

### 8. Temporal Localization

The project also investigates temporal localization of manipulated content.

Temporal localization components analyze video segments and predicted temporal intervals to identify where potential manipulations occur. BMN-based experiments, temporal evaluation and localization comparisons are included in the research pipeline.

## Evaluation and Analysis

The framework includes evaluation and analysis tools for:

- Classification performance
- Individual agent comparison
- Fusion strategy comparison
- Reliability analysis
- Calibration analysis
- Decision threshold analysis
- Modality ablation
- Prediction uncertainty
- Cross-dataset comparison
- Temporal localization
- Temporal alignment and reranking
- Error and evidence analysis

Evaluation scripts and result-generation utilities are provided within the project code.

## Project Structure

```text
ics-av-deepfake_team_share/
│
├── baseline_final/
│   └── Baseline implementation
│
├── model/
│   └── Model components
│
├── proposed_method/
│   └── Proposed methodology implementation
│
├── proposed_model/
│   ├── Model implementation
│   ├── Agent components
│   ├── Fusion mechanisms
│   ├── Training and evaluation code
│   ├── Calibration and uncertainty analysis
│   ├── Temporal localization components
│   ├── checkpoints/
│   └── outputs/
│       └── Evaluation result files
│
├── decoder_voxceleb/
│   └── VoxCeleb decoding utilities
│
├── decoder_voxceleb_feat/
│   └── Feature decoding utilities
│
├── nms_utils/
│   └── Non-maximum suppression utilities
│
├── documentation/
│   └── Project documentation
│
├── README.md
├── requirements.txt
└── .gitignore
```

*The directory structure is representative of the major project components. Some scripts and supporting utilities are maintained in the root directory and other subdirectories.*

## Model Checkpoints

Selected trained model checkpoints are provided for the proposed framework, subject to repository file-size limitations.

The checkpoints include:

- `proposed_best.pt` — checkpoint for the proposed model.
- `proposed_reliability_best.pt` — checkpoint associated with the reliability-based model.
- `proposed_temporal_best.pt` — checkpoint associated with the temporal model.

These checkpoints can be used with the corresponding model implementation and inference pipeline.

## Results and Experiment Artifacts

The project maintains experiment outputs in JSON format, including:

- Agent comparison
- Proposed model evaluation
- Reliability evaluation
- Fusion comparison
- Calibration audits
- Modality ablation
- Evidence analysis
- Uncertainty evaluation
- Validation threshold analysis
- Temporal alignment
- Temporal reranking comparison

These artifacts support experimental analysis and reproducibility without requiring all experiments to be executed again.

## Installation

### 1. Clone the repository

```bash
git clone <https://github.com/Nisha-Sharma676/An-Intelligent-Multi-Agent-Framework-for-Trustworthy-Synthetic-Media-Detection>
cd ics-av-deepfake_team_share
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

The repository includes scripts for model training, inference, evaluation, feature extraction, agent comparison, fusion analysis, calibration, uncertainty analysis and temporal localization.

The relevant scripts can be executed according to the configured dataset paths, feature locations, model checkpoints and experiment settings.

Before running any script:

1. Prepare the required dataset.
2. Configure the dataset and feature paths.
3. Verify the required model checkpoints.
4. Install the required dependencies.
5. Run the corresponding training, inference or evaluation script.

Exact commands may vary depending on the selected experiment and configuration.

## Reproducibility

The project includes saved model checkpoints, experiment result artifacts, evaluation scripts and analysis utilities to support reproducibility and further investigation.

Reproducing an experiment may require the original datasets, extracted features, compatible software dependencies, configured paths and the corresponding checkpoint.

## Limitations

- Detection performance may vary across datasets, manipulation methods and recording conditions.
- Individual agents may behave differently depending on the available modalities and input quality.
- Prediction confidence does not always guarantee prediction correctness.
- Temporal localization remains challenging, particularly when manipulated segments are short or difficult to identify.
- Cross-dataset generalization may be affected by differences in data distribution and manipulation techniques.

## Future Work

- Improve agent-level reliability and fusion strategies.
- Strengthen cross-dataset generalization.
- Improve uncertainty estimation and calibration.
- Refine temporal localization and manipulated-segment detection.
- Investigate robustness against unseen manipulation techniques.
- Improve reproducibility and deployment efficiency.

## Acknowledgements

This project explores multi-agent audio-visual deepfake detection, combining classification, agent-level analysis, fusion, reliability assessment, uncertainty analysis and temporal localization within a unified research framework.
