# Similarity Equivariant MLP

This repository contains layers of similarity equivariant MLPs based on [Huang2022] and [Horie2024].
It confirms that the model has the scaling equivariance during and after training using synthetic data.

# Usage
```bash
poetry run python3 main.py
```

# References
- [Huang2022] W. Huang et al. [Equivariant Graph Mechanics Networks with Constraints](https://arxiv.org/abs/2203.06442). ICLR 2022.
- [Horie2024] M. Horie and N. Mitsume. [Graph Neural PDE Solvers with Conservation and Similarity-Equivariance](https://arxiv.org/abs/2405.16183). ICML 2024.
