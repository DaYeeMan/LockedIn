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

- [ ] 8. Implement MDA-CNN model architecture
  - Create CNN branch for processing LF surface patches
  - Implement MLP branch for point feature processing
  - Add fusion layer to combine CNN and MLP representations
  - Implement residual prediction head with appropriate activation
  - Write unit tests for model component functionality
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 9. Implement baseline models for comparison
  - Create exact Funahashi baseline model (5 layers, 32 neurons, ReLU, residual learning)
  - Implement direct MLP model (point features → volatility)
  - Create residual MLP model (point features → residual, no patches)
  - Add simple CNN-only model for ablation studies
  - Ensure consistent interfaces across all model architectures
  - Write tests for baseline model training and inference
  - _Requirements: 3.1, 6.1_

- [ ] 10. Create training infrastructure and loss functions
  - Implement main training loop with validation and early stopping
  - Add custom loss functions (MSE, weighted MSE for wings)
  - Create model checkpointing and best model saving
  - Implement learning rate scheduling and gradient clipping
  - Add training progress monitoring and logging
  - _Requirements: 2.5, 3.3_

- [ ] 11. Implement comprehensive evaluation metrics
  - Create Funahashi's exact metrics (MSE, relative percentage error)
  - Add surface-specific evaluation metrics (RMSE, MAE, relative error)
  - Implement region-specific metrics for ATM, ITM, OTM performance analysis with emphasis on deep wing regions
  - Create wing error analysis where MC-Hagan residuals are largest (moneyness < 0.7 or > 1.3)
  - Create direct comparison pipeline with Funahashi's published results
  - Implement statistical significance testing for model comparisons
  - Create evaluation pipeline that works across different HF budgets
  - Write tests for metric calculation accuracy
  - _Requirements: 3.3, 6.1, 6.3_

- [ ] 12. Create experiment orchestrator for HF budget analysis
  - Implement experiment runner that tests multiple HF budget sizes
  - Add automated model comparison across different architectures
  - Create results aggregation and statistical analysis
  - Implement automated hyperparameter tuning for each budget
  - Add experiment reproducibility with proper seed management
  - _Requirements: 3.2, 5.2_

- [ ] 13. Implement volatility smile visualization tools
  - Create smile plotting function comparing HF, LF, baseline, and MDA-CNN predictions
  - Add support for multiple parameter sets and market conditions
  - Implement error visualization with confidence intervals
  - Create interactive plots for detailed analysis
  - Write tests for plot generation and data accuracy
  - _Requirements: 4.1_

- [ ] 14. Create 3D surface visualization and analysis tools
  - Implement 3D surface plotting for volatility surfaces
  - Add error heatmap overlays showing prediction accuracy
  - Create surface difference plots (predicted vs actual)
  - Add support for multiple surface comparisons in single plot
  - Write tests for visualization data consistency
  - _Requirements: 4.3_

- [ ] 15. Implement performance analysis and reporting
  - Create performance vs HF budget analysis plots
  - Implement residual distribution analysis before/after ML correction
  - Add training convergence visualization and analysis
  - Create automated report generation with key metrics and plots
  - Write comprehensive evaluation pipeline that generates all analysis
  - _Requirements: 4.2, 4.4, 4.5_

- [ ] 16. Create main execution scripts and user interface
  - Implement main data generation script with command-line interface
  - Create main training script with configurable experiments
  - Add evaluation and visualization script for results analysis
  - Create example notebooks demonstrating full workflow
  - Add comprehensive documentation and usage examples
  - _Requirements: 5.4_

- [ ] 17. Implement performance optimizations
  - Add GPU acceleration for model training and inference
  - Optimize data loading with prefetching and parallel processing
  - Implement memory-efficient batch processing for large datasets
  - Add computational profiling and bottleneck identification
  - Write performance benchmarks and optimization tests
  - _Requirements: 6.2, 6.3, 6.5_

- [ ] 18. Create comprehensive test suite and validation
  - Implement end-to-end integration tests for complete pipeline
  - Add financial validation tests against known SABR solutions
  - Create performance regression tests for computational efficiency
  - Implement data consistency tests across pipeline stages
  - Add model convergence and stability tests
  - _Requirements: 5.2, 5.5_

- [ ] 19. Generate research comparison and publication materials
  - Create comprehensive comparison study between MDA-CNN and Funahashi baseline
  - Generate research-quality plots comparing performance vs HF budget
  - Produce statistical analysis tables suitable for academic publication
  - Create data efficiency analysis showing MDA-CNN advantages
  - Generate detailed wing performance comparison plots showing MDA-CNN advantages in deep ITM/OTM regions
  - Create residual magnitude analysis showing where MDA-CNN patches provide most benefit
  - Create summary tables matching Funahashi's paper format for direct comparison
  - _Requirements: 6.3, 6.4, 6.5_

- [ ] 20. Final integration and documentation
  - Integrate all components into cohesive system
  - Create comprehensive README with setup and usage instructions
  - Add example configuration files for different experiment types
  - Implement error handling and user-friendly error messages
  - Create final validation run with complete workflow demonstration
  - _Requirements: 5.1, 5.4_