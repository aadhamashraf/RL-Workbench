## RL Playground

* Download and install [Anaconda](https://www.anaconda.com/products/distribution)

* Restart the terminal

* Verify Conda works

```bash
conda --version
```

* You should see something like:

```
conda 24.x.x
```

* Create a new conda environment

```bash
conda create -n rl python=3.10 -y
conda activate rl
```

* Install gymnasium and dependencies

```bash
conda install -c conda-forge gymnasium box2d-py pygame
```

* Verify gymnasium works

```bash
gymnasium --version
```

* You should see something like:

```
gymnasium 0.x.x
```
