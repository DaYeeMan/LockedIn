# Implementation Plan

- [x] 1. Set up project structure and core utilities





  - Implement configuration management system for experiments and hyperparameters
  - Create logging utilities and random seed management for reproducibility
  - _Requirements: 5.1, 5.3, 5.5_

- [x] 2. Implement SABR parameter and grid configuration classes





  - Create SABRParams dataclass with Funahashi's parameter ranges and validation
  - Implement GridConfig class for surface discretization with 21 strikes per Funahashi
  - Add parameter sampling strategies matching Funahashi's approach
  - Implement Funahashi's strike range calculation: K1 = max(f - 1.8√V, 0.4f), K2 = min(f + 2√V, 2f)
  - Add extended wing sampling for research: additional strikes at 0.3f and 2.5f for deep wing analysis
  - Write unit tests for parameter validation and sampling
  - _Requirements: 1.3, 6.1_

- [x] 3. Implement Monte Carlo SABR simulation engine





  - Create SABR Monte Carlo path generator using log-Euler scheme
  - Implement volatility surface calculation from MC paths
  - Add parallel processing support for multiple parameter sets
  - Include convergence checks and numerical stability handling
  - Write tests for MC accuracy against known analytical cases
  - _Requirements: 1.1, 6.1, 6.4_

- [x] 4. Implement Hagan analytical surface generator





  - Create Hagan formula implementation for SABR volatility surfaces
  - Handle edge cases and numerical stability in Hagan approximation
  - Implement efficient vectorized surface evaluation across strike/maturity grids
  - Add validation against literature benchmarks
  - Write unit tests for Hagan formula accuracy
  - _Requirements: 1.2, 1.3_

- [x] 5. Create data generation orchestrator and validation








  - Implement main data generation pipeline that coordinates MC and Hagan surface creation
  - Add data quality validation and outlier detection
  - Create data saving/loading utilities with proper file organization
  - Implement progress tracking and estimated completion times
  - Write integration tests for complete data generation workflow
  - _Requirements: 1.4, 6.4_

- [x] 5.1. Implement data visualization for generated surfaces







  - Create visualization tools for MC and Hagan surface comparison
  - Implement 3D surface plots showing raw MC vs Hagan surfaces
  - Add residual heatmaps showing D(ξ) = σ_MC(ξ) - σ_Hagan(ξ) across parameter space
  - Create volatility smile plots for individual parameter sets
  - Add statistical distribution plots of residuals to identify wing regions
  - Implement interactive plots for exploring parameter space effects
  - _Requirements: 4.1, 4.3_

- [x] 6. Implement patch extraction and feature engineering





  - Create PatchExtractor class to extract local surface patches around HF points
  - Implement grid alignment logic to map HF points to LF surface coordinates
  - Create FeatureEngineer class for point feature creation and normalization
  - Add support for different patch sizes and boundary handling
  - Write tests for patch extraction accuracy and feature normalization
  - _Requirements: 2.2, 2.3_

- [x] 7. Implement data preprocessing and loading pipeline





  - Create efficient data loader with batching and shuffling capabilities
  - Implement data normalization and scaling utilities
  - Add support for train/validation/test splits with proper indexing
  - Create HDF5-based storage for preprocessed training data
  - Write tests for data loading consistency and performance
  - _Requirements: 6.3, 6.5_

- [x] 8. Implement MDA-CNN model architecture




  - Create CNN branch for processing LF surface patches
  - Implement MLP branch for point feature processing
  - Add fusion layer to combine CNN and MLP representations
  - Implement residual prediction head with appropriate activation
  - Write unit tests for model component functionality
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 9. Implement baseline models for comparison






  - Create exact Funahashi baseline model (5 layers, 32 neurons, ReLU, residual learning)
  - Implement direct MLP model (point features → volatility)
  - Create residual MLP model (point features → residual, no patches)
  - Add simple CNN-only model for ablation studies
  - Ensure consistent interfaces across all model architectures
  - Write tests for baseline model training and inference
  - _Requirements: 3.1, 6.1_

- [x] 10. Create training infrastructure and loss functions





  - Implement main training loop with validation and early stopping
  - Add MSE loss function for residual learning
  - Create model checkpointing and best model saving
  - Add training progress monitoring and logging
  - _Requirements: 2.5, 3.3_

- [x] 11. Implement evaluation metrics and model comparison





  - Create Funahashi's exact metrics (MSE, relative percentage error, RMSE, MAE)
  - Implement direct comparison pipeline between MDA-CNN and Funahashi baseline
  - Create evaluation pipeline that compares both models using same HF budget
  - Generate comparison tables matching Funahashi's paper format
  - _Requirements: 3.3, 6.1_

- [x] 12. Create surface visualization and comparison tools




  - Implement 3D surface plotting for MC, Hagan, baseline, and MDA-CNN surfaces
  - Create volatility smile plots comparing all four approaches
  - Add error visualization showing prediction accuracy differences
  - Create surface difference plots (predicted vs actual) for both models
  - Generate side-by-side comparison plots for direct analysis
  - _Requirements: 4.1, 4.3_

- [x] 13. Create main execution and comparison scripts






  - Implement main training script that trains both models with same data
  - Create evaluation script that compares both models and generates all plots
  - Add configuration management for experiment parameters
  - Create final comparison report with all visualizations and metrics
  - _Requirements: 5.4, 6.1_