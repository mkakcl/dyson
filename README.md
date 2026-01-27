# `dyson`: Dyson equation solvers for Green's function methods

[![CI](https://github.com/BoothGroup/dyson/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/BoothGroup/dyson/actions/workflows/ci.yaml)
[![ruff](https://camo.githubusercontent.com/530951ce7a8a18468f6644ee7fd065389265c4b7176fce55f271b283be3a83c6/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f636861726c6965726d617273682f727566662f6d61696e2f6173736574732f62616467652f76312e6a736f6e)](https://github.com/astral-sh/ruff)

The `dyson` package implements various Dyson equation solvers, including novel approaches that avoiding explicitly grid-resolved numerical procedures such as Fourier transforms and analytical continuation.
These include the moment-resolved block Lanczos methods for moments of the Green's function or self-energy.

## Installation

From source:

```bash
git clone https://github.com/BoothGroup/dyson
pip install .
```

## Usage

Examples are available in the `examples` directory.
