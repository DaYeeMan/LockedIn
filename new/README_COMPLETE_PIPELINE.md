# SABR Volatility Surface Modeling - Complete Pipeline

This is the complete implementation of the SABR volatility surface modeling project comparing MDA-CNN with Funahashi's baseline model.

## 🚀 Quick Start

### 1. Check Pipeline Status
```bash
cd new
python check_pipeline.py
```
This verifies all components are properly set up.

### 2. Test Data Generation (Recommended)
```bash
python test_data_generation.py
```
Quick test with minimal data to verify everything works.

### 3. Run Complete Experiment
```bash
python run_experiment.py
```
This runs the entire pipeline: data generation → training → evaluation.

## 📋 Pipeline Overview

The complete pipeline consists of three main phases:

### Phase 1: Data Generation
- **Script**: `generate_training_data.py`
- **Purpose**: Generate SABR volatility surfaces and training data
- **Output**: Raw surfaces + preprocessed training samples
- **Time**: ~10-30 minutes (depending on configuration)

### Phase 2: Model Training  
- **Script**: `main_training.py`
- **Purpose**: Train both MDA-CNN and Funahashi baseline models
- **Output**: Trained models + training logs
- **Time**: ~30-60 minutes (depending on data size and hardware)

### Phase 3: Model Evaluation
- **Script**: `main_evaluation.py` 
- **Purpose**: Comprehensive comparison and analysis
- **Output**: Metrics, plots, and comparison report
- **Time**: ~5-10 minutes

## 🔧 Individual Components

### Data Generation
```bash
# Generate with default settings (1000 parameter sets)
python generate_training_data.py

# Generate with custom settings
python generate_training_data.py --n-parameter-sets 2000 --create-visualizations

# Use custom config
python generate_training_data.py --config my_data_config.yaml
```

### Training
```bash
# Train with generated data
python main_training.py --data-dir data/processed

# Train with custom config
python main_training.py --config config/training_config.yaml --data-dir data/processed
```

### Evaluation
```bash
# Evaluate trained models
python main_evaluation.py --experiment-dir results/sabr_comparison_20241106_143022

# Detailed analysis with extra plots
python main_evaluation.py --experiment-dir results/[experiment] --detailed-analysis
```

## ⚙️ Configuration

### Data Generation Config (`config/data_generation_config.yaml`)
```yaml
n_parameter_sets: 1000          # Number of SABR parameter sets
mc_paths: 100000               # Monte Carlo paths per surface
sampling_strategy: "lhs"        # Parameter sampling method
patch_size: 9                  # CNN patch size
parallel_processing: true      # Use multiprocessing
```

### Training Config (`config/training_config.yaml`)
```yaml
experiment:
  batch_size: 64
  learning_rate: 0.0003
  epochs: 200
  patch_size: [9, 9]

model:
  cnn_filters: [32, 64, 128]
  mlp_hidden_dims: [64, 64]
  fusion_dim: 128
  dropout_rate: 0.2
```

## 📊 Expected Results

### Data Generation
- **1,000 SABR parameter sets** (Latin Hypercube Sampling)
- **~20,000-50,000 training samples** (extracted from surfaces)
- **Quality validation** with automatic outlier detection
- **Data splits**: 70% train, 15% validation, 15% test

### Model Training
- **MDA-CNN**: Multi-fidelity CNN with patch + point features
- **Funahashi Baseline**: 5-layer MLP (32 neurons, ReLU, residual learning)
- **Same data budget** for fair comparison
- **Training metrics**: MSE loss, validation curves

### Model Evaluation
- **Performance metrics**: MSE, RMSE, MAE, relative error
- **Surface visualizations**: 3D plots, volatility smiles
- **Error analysis**: Residual distributions, error maps
- **Direct comparison** with Funahashi's published results

## 🗂️ Output Structure

```
├── data/
│   ├── raw/                    # Generated surfaces
│   └── processed/              # Training-ready data
├── results/
│   └── sabr_comparison_[timestamp]/
│       ├── mda_cnn/           # MDA-CNN model files
│       ├── funahashi/         # Funahashi model files
│       └── evaluation/        # Comparison results
└── config/                     # Configuration files
```

## 🎯 Requirements Compliance

This implementation satisfies all specification requirements:

- ✅ **Requirement 1**: Complete data generation infrastructure
- ✅ **Requirement 2**: MDA-CNN architecture with CNN + MLP branches  
- ✅ **Requirement 3**: Training framework with Funahashi baseline
- ✅ **Requirement 4**: Comprehensive visualization and comparison
- ✅ **Requirement 5**: Organized project structure and reproducibility
- ✅ **Requirement 6**: Direct model comparison with same HF data budget

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd new
   python check_pipeline.py
   ```

2. **Memory Issues**
   ```bash
   # Reduce data generation size
   python generate_training_data.py --n-parameter-sets 100
   ```

3. **Missing Dependencies**
   ```bash
   pip install torch numpy scipy h5py pyyaml matplotlib
   ```

4. **Data Not Found**
   ```bash
   # Generate data first
   python generate_training_data.py
   # Then run training
   python main_training.py --data-dir data/processed
   ```

### Performance Tips

- **Faster Generation**: Enable parallel processing, reduce MC paths
- **Better Quality**: Increase parameter sets, use LHS sampling
- **GPU Training**: Ensure PyTorch CUDA is installed
- **Memory Optimization**: Reduce batch size if needed

## 📈 Customization

### Different Parameter Ranges
Edit `config/data_generation_config.yaml` to modify SABR parameter ranges.

### Model Architecture Changes
Edit `config/training_config.yaml` to modify CNN/MLP architectures.

### Additional Baselines
Add new models to `models/baseline_models.py` and update training scripts.

### Custom Metrics
Add new evaluation metrics to `evaluation/metrics.py`.

## 🧪 Testing

### Quick Tests
```bash
python test_data_generation.py     # Test data generation
python check_pipeline.py           # Check all components
```

### Integration Tests
```bash
# Test with minimal data
python run_experiment.py --generate-data-only
python run_experiment.py --skip-data-generation
```

## 📚 Documentation

- **Data Generation**: `README_data_generation.md`
- **Execution Guide**: `README_execution.md`
- **Model Architecture**: See `models/` directory
- **Evaluation Metrics**: See `evaluation/` directory

## 🎉 Success Indicators

When everything works correctly, you should see:

1. **Data Generation**: Quality score > 0.8, validation passes
2. **Training**: Both models train successfully, validation loss decreases
3. **Evaluation**: Comprehensive comparison plots and metrics
4. **Results**: Direct comparison with Funahashi's published results

The pipeline is designed to be robust, reproducible, and directly comparable to the Funahashi baseline for fair performance evaluation.