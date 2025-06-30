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

## Acknowledgments

We sincerely thank Zhou et al. for their outstanding contribution to the multimodal research community through their open-source MMRec framework. Their work has significantly advanced the field of multimodal recommendation systems.

- MMRec GitHub: [https://github.com/enoche/MMRec](https://github.com/enoche/MMRec)
- MMRec Citation:
```bibtex
@inproceedings{zhou2023bootstrap,
  author = {Zhou, Xin and Zhou, Hongyu and Liu, Yong and Zeng, Zhiwei and Miao, Chunyan and Wang, Pengwei and You, Yuan and Jiang, Feijun},
  title = {Bootstrap Latent Representations for Multi-Modal Recommendation},
  booktitle = {Proceedings of the ACM Web Conference 2023},
  pages = {845–854},
  year = {2023}
}
```

## Citation

If you find this framework useful for your research, please consider citing our paper:
```bibtex
[Citation information will be added here]
```