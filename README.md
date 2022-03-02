# Manifold-constrained Gaussian process inference for time-varying parameters in dynamic systems

This repository provides the PyTorch implementation of TVMAGI, an extension of the MAnifold-constrained Gaussian process Inference (MAGI) for learning time-varying parameters from a perturbed dynamic system.

## TVMAGI codes and requirements
The codes for TVMAGI are provided under directory TVMAGI/. To successfully run the TVMAGI code, we require the installation of the following python packages. We provide the version that we use, but other version of the packages is also allowed as long as it is compatible.

```
pip3 install numpy==1.19.5 scipy==1.6.1 torch==1.8.0 matplotlib==3.3.4
docker run -d -p shihaoyangphd/magi:map
```

Please see demo.ipynb and three examples in Examples/. for tutorial of running TVMAGI.

References Our paper is available on [arXiv](https://arxiv.org/abs/2105.13407). If you found this repository useful in your research, please consider citing

```
@misc{sun2022manifoldconstrained,
      title={Manifold-constrained Gaussian process inference for time-varying parameters in dynamic systems}, 
      author={Yan Sun and Shihao Yang},
      year={2022},
      eprint={2105.13407},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```
