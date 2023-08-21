<a href="https://github.com/NiMlr/pynlqn">
  <img align="right" width="150" height="150" src="https://github.com/mdp-toolkit/mdp-toolkit/assets/39880630/9795e7ca-35aa-4191-bd07-60e10b5438c1"><br>
</a>


# pynlqn - Global Optmization with Gradients.

**pynlqn** is a `Scipy`-based Python implementation of an experimental
non-local Newton method.

- [**Installation**](https://github.com/NiMlr/pynlqn#installation)
- [**Usage Tutorial**](https://github.com/NiMlr/pynlqn#usage-tutorial)
- [**How to Cite**](https://github.com/NiMlr/pynlqn#how-to-cite)

### Installation

`pynlqn` can be installed directly from source with:
```sh
pip install git+https://github.com/NiMlr/pynlqn
```


### Usage Tutorial

Try to optimize a function with many suboptimal local minima.
```python
import pynlqn
import numpy as np

# set dimension and initial point
n = 10
x0 = 5.*np.ones(n)

# define Rastrigin-type function and gradient
f = lambda x: 2.*n + np.sum(x**2 - 2.*np.cos(2.*np.pi*x), axis=0)
gf = lambda x: 2*x + 4.*np.pi*np.sin(2.*np.pi*x)

# define algorithm parameters
C = 10**4 # budget
sigma0 = 100. # initial scaling
k = 100 # sample points per iteration

# fix a seed
np.random.seed(0)

# run the optimization method nlqn to find global minimum at 0
pynlqn.nlqn(f, gf, x0, sigma0, k, C, verbose=False)
# array([ 3.24988921e-09, -2.58312173e-10, -9.27461395e-10, -1.37223679e-09,
#         5.23734887e-10, -8.15789773e-10,  1.82301168e-10,  7.31047982e-10,
#        -7.45285151e-10, -6.81223695e-10])
```


### How to Cite

If you use `pynlqn` in your research, please consider citing the associated [paper](https://arxiv.org/abs/2308.09556).

```bibtex
@misc{müller2023principle,
      title={A Principle for Global Optimization with Gradients}, 
      author={Nils Müller},
      year={2023},
      eprint={2308.09556},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```
