# Requirements Document

## Introduction

This project aims to develop a Multi-fidelity Data Aggregation CNN (MDA-CNN) architecture for SABR volatility surface modeling and compare it directly against Funahashi's "SABR Equipped with AI Wings" baseline model. The system will generate both high-fidelity Monte Carlo simulations and low-fidelity Hagan analytical surfaces, then train both MDA-CNN and Funahashi baseline models to predict residuals between MC and Hagan surfaces using the same HF data budget. The goal is to demonstrate whether MDA-CNN can achieve superior accuracy by leveraging local surface patches, with direct comparison to published Funahashi results.

## Requirements

### Requirement 1: Data Generation Infrastructure

**User Story:** As a quantitative researcher, I want to generate comprehensive SABR volatility surface data using both Monte Carlo simulations and Hagan analytical formulas, so that I can train multi-fidelity models with sufficient data coverage.

#### Acceptance Criteria

1. WHEN the system generates data THEN it SHALL create high-fidelity Monte Carlo volatility surfaces with configurable number of simulation paths
2. WHEN the system generates data THEN it SHALL create low-fidelity Hagan analytical volatility surfaces covering the same parameter space
3. WHEN generating surfaces THEN the system SHALL support Funahashi's parameter ranges (α: 0.05-0.6, β: 0.3-0.9, ν: 0.05-0.9, ρ: -0.75-0.75, T: 1-10 years) for direct comparison
4. WHEN data generation is complete THEN the system SHALL save datasets in organized folder structure with clear naming conventions
5. WHEN data is generated THEN the system SHALL provide visualization tools to inspect MC and Hagan surfaces, residuals, and parameter space coverage
6. IF the user specifies a limited number of HF points THEN the system SHALL strategically sample across the parameter space to maximize coverage

### Requirement 2: Multi-fidelity Model Architecture

**User Story:** As a machine learning engineer, I want to implement an MDA-CNN architecture that leverages both local surface patches and point features, so that I can predict volatility surface residuals with high accuracy using minimal high-fidelity data.

#### Acceptance Criteria

1. WHEN building the model THEN the system SHALL implement a CNN branch that processes local LF surface patches (e.g., 9x9 grids)
2. WHEN building the model THEN the system SHALL implement an MLP branch that processes point features (SABR parameters, strike, maturity)
3. WHEN processing inputs THEN the system SHALL concatenate CNN and MLP latent representations before final prediction
4. WHEN making predictions THEN the model SHALL output residual values D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)
5. WHEN training THEN the system SHALL use appropriate loss functions (MSE) and regularization techniques

### Requirement 3: Training and Evaluation Framework

**User Story:** As a researcher, I want to train and evaluate both MDA-CNN and Funahashi baseline models using the same HF data budget, so that I can make a direct comparison of their performance.

#### Acceptance Criteria

1. WHEN training models THEN the system SHALL implement Funahashi's exact baseline model (5-layer, 32 neurons, ReLU activation, residual learning)
2. WHEN evaluating models THEN the system SHALL train both models using the same fixed HF data budget
3. WHEN computing metrics THEN the system SHALL calculate MSE, relative percentage error, RMSE, and MAE to enable direct comparison with Funahashi's published results
4. WHEN training is complete THEN the system SHALL save model checkpoints and training logs

### Requirement 4: Visualization and Comparison

**User Story:** As a quantitative analyst, I want to visualize generated surfaces and compare model errors directly against Funahashi's published results, so that I can validate the effectiveness of the MDA-CNN approach.

#### Acceptance Criteria

1. WHEN generating visualizations THEN the system SHALL plot volatility surfaces comparing MC, Hagan, Funahashi baseline, and MDA-CNN predictions
2. WHEN displaying results THEN the system SHALL create error comparison plots showing MDA-CNN vs Funahashi baseline performance
3. WHEN visualizing surfaces THEN the system SHALL generate 3D surface plots and volatility smiles for different parameter combinations
4. WHEN presenting results THEN the system SHALL generate comparison tables matching Funahashi's paper format for direct validation

### Requirement 5: Project Organization and Reproducibility

**User Story:** As a developer, I want a well-organized project structure with reproducible experiments and clear documentation, so that the research can be easily understood, extended, and replicated.

#### Acceptance Criteria

1. WHEN organizing the project THEN the system SHALL create separate folders for data generation, model training, evaluation, and visualization
2. WHEN running experiments THEN the system SHALL use fixed random seeds for reproducible results
3. WHEN saving outputs THEN the system SHALL organize results in clearly labeled directories with timestamps
4. WHEN documenting code THEN the system SHALL include comprehensive docstrings and comments
5. WHEN providing configuration THEN the system SHALL use configuration files for hyperparameters and experimental settings

### Requirement 6: Direct Model Comparison

**User Story:** As a researcher, I want to compare MDA-CNN directly against Funahashi's baseline using the same HF data budget, so that I can determine which approach achieves better accuracy.

#### Acceptance Criteria

1. WHEN comparing models THEN the system SHALL train both MDA-CNN and Funahashi baseline on identical parameter spaces and datasets with the same HF budget
2. WHEN generating results THEN the system SHALL produce comparison tables matching Funahashi's paper format for direct result validation
3. WHEN documenting results THEN the system SHALL generate error comparison plots and surface visualizations suitable for analysis

