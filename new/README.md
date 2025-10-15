# SABR Volatility Surface MDA-CNN

Multi-fidelity Data Aggregation CNN for SABR volatility surface modeling that outperforms existing approaches by leveraging both high-fidelity Monte Carlo simulations and low-fidelity Hagan analytical surfaces.

## Project Overview

This project implements a novel MDA-CNN architecture that:
- Combines CNN processing of local surface patches with MLP processing of point features
- Predicts residuals between Monte Carlo and Hagan surfaces
- Achieves superior accuracy in wing regions (deep ITM/OTM) with fewer high-fidelity data points
- Provides comprehensive comparison with Funahashi's "SABR Equipped with AI Wings" baseline

## Project Structure

```
new/
├── config/                 # Configuration management
│   ├── config_manager.py   # Main configuration system
│   └── default_config.yaml # Default experiment settings
├── utils/                  # Core utilities
│   ├── logging_utils.py    # Structured logging system
│   ├── seed_utils.py       # Reproducibility management
│   └── experiment_utils.py # Experiment orchestration
├── data_generation/        # Data generation modules
├── models/                 # Model architectures
├── preprocessing/          # Data preprocessing
├── training/              # Training infrastructure
├── evaluation/            # Evaluation metrics and analysis
├── visualization/         # Plotting and visualization
├── examples/              # Example scripts and notebooks
└── requirements.txt       # Python dependencies
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run basic setup example:**
   ```bash
   python examples/basic_setup_example.py
   ```

3. **Start with default configuration:**
   ```python
   from utils.experiment_utils import create_experiment
   
   with create_experiment("my_experiment") as experiment:
       logger = experiment.get_logger()
       config = experiment.get_config()
       logger.info("Experiment started!")
   ```

## Key Features

### Configuration Management
- YAML/JSON configuration files with validation
- Environment variable override support
- Hierarchical configuration structure (experiment, model, data)
- Default configurations matching Funahashi's paper parameters

### Reproducibility
- Global random seed management across all libraries (NumPy, TensorFlow, PyTorch)
- Deterministic behavior configuration
- Reproducible data splits and experiment runs
- Context managers for temporary seed changes

### Logging
- Structured logging with multiple output formats
- Experiment tracking with metrics and hyperparameters
- Colored console output and JSON file logging
- Automatic log rotation and organization

### Experiment Management
- Complete experiment lifecycle management
- Automatic output directory organization
- Artifact saving and loading
- Configuration persistence and versioning

## Configuration

The system uses a hierarchical configuration structure:

```yaml
experiment:
  name: "sabr_mdacnn_experiment"
  random_seed: 42
  hf_budget: 200
  # ... other experiment settings

model:
  model_type: "mda_cnn"
  patch_size: [9, 9]
  # ... model architecture settings

data:
  data_dir: "data"
  normalize_features: true
  # ... data processing settings
```

## Research Goals

This implementation aims to demonstrate:
1. **Superior Data Efficiency**: MDA-CNN achieves better performance with fewer HF points than Funahashi's baseline
2. **Wing Region Accuracy**: Improved performance in deep ITM/OTM regions where MC-Hagan residuals are largest
3. **Spatial Pattern Learning**: CNN patches capture local volatility surface patterns that point-wise methods miss
4. **Comprehensive Comparison**: Direct comparison with Funahashi's published results using identical parameter spaces

## Next Steps

After completing this setup task, the implementation plan continues with:
1. SABR parameter and grid configuration classes
2. Monte Carlo simulation engine
3. Hagan analytical surface generator
4. Data generation orchestrator
5. MDA-CNN model architecture
6. Training and evaluation infrastructure

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn for visualization
- PyYAML for configuration
- See `requirements.txt` for complete list

## License

This project is developed for research purposes. Please cite appropriately if used in academic work.