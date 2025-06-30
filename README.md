# ESDvr: Exploring the Synergetic and Divergent Potentials of Multimodal Semantics for Feature Fusion-based Video Recommendation

This repository contains the implementation of ESDvr, a novel video recommendation system that explores the synergetic and divergent potentials of multimodal semantics through feature fusion.

## Repository Structure

- `./data`: Contains all dataset-related files
  - Due to upload size limitations, we provide:
    - Original dataset sources/links
    - All preprocessing and web crawling scripts
    - Processed partial data from Movielens dataset
- `./src/models/try.py`: Our main model implementation
- `./src/main.py`: Main entry point to run the system

## How to Run

To execute our model:
```bash
python3 ./src/main.py -c log_name
```

## Requirements

- Environment specifications are listed in `requirements.txt`
- Our GPU configuration:
  - NVIDIA-SMI: 550.54.14
  - Driver Version: 550.54.14
  - CUDA Version: 12.4

## Citation

If you find this framework useful for your research, please consider citing our paper:
```bibtex
[Citation information will be added here]
```