
#!/bin/bash
CONDA_PATH_INSTALL=/lustre/fswork/projects/rech/soz/commun/IAlefeu/conda_wigs

#!/bin/bash
module purge

module load arch/h100

# Load miniforge
module load miniforge/24.11.3

# Load baseline env
conda activate pytorch-gpu-2.8.0+py3.12.11

# Activate env with --stack
conda activate --stack $CONDA_PATH_INSTALL

# Load dependancies
module load pytorch-gpu/py3/2.8.0

# Change PYTHONUSERBASE to be able to use bin in the env conda_path
export PYTHONUSERBASE=$CONDA_PATH_INSTALL

SLURM_ARRAY_TASK_ID=0

```
>>> import torch
>>> 
```



<!-- --- 

#!/bin/bash
module purge


# Load dependancies
module load pytorch-gpu/py3/2.8.0


# Load miniforge
module load miniforge/24.11.3

# Load baseline env
conda activate pytorch-gpu-2.8.0+py3.12.11


# Activate env with --stack
conda activate --stack $CONDA_PATH_INSTALL

# Change PYTHONUSERBASE to be able to use bin in the env conda_path
export PYTHONUSERBASE=$CONDA_PATH_INSTALL


module load arch/h100

```
>>> import torch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/lustre/fshomisc/sup/hpe/pub/miniforge/24.11.3/envs/pytorch-gpu-2.8.0+py3.12.11/lib/python3.12/site-packages/torch/__init__.py", line 415, in <module>
    _load_global_deps()
  File "/lustre/fshomisc/sup/hpe/pub/miniforge/24.11.3/envs/pytorch-gpu-2.8.0+py3.12.11/lib/python3.12/site-packages/torch/__init__.py", line 371, in _load_global_deps
    raise err
  File "/lustre/fshomisc/sup/hpe/pub/miniforge/24.11.3/envs/pytorch-gpu-2.8.0+py3.12.11/lib/python3.12/site-packages/torch/__init__.py", line 320, in _load_global_deps
    ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)
  File "/lustre/fshomisc/sup/hpe/pub/miniforge/24.11.3/envs/pytorch-gpu-2.8.0+py3.12.11/lib/python3.12/ctypes/__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: libmpi.so.40: cannot open shared object file: No such file or directory
>>> 
```

---




---

#!/bin/bash
CONDA_PATH_INSTALL=/lustre/fswork/projects/rech/soz/commun/IAlefeu/conda_wigs

#!/bin/bash
module purge



module load arch/h100

# Load baseline env
conda activate pytorch-gpu-2.8.0+py3.12.11

# Load miniforge
module load miniforge/24.11.3


# Activate env with --stack
conda activate --stack $CONDA_PATH_INSTALL

# Load dependancies
module load pytorch-gpu/py3/2.8.0

# Change PYTHONUSERBASE to be able to use bin in the env conda_path
export PYTHONUSERBASE=$CONDA_PATH_INSTALL

SLURM_ARRAY_TASK_ID=0

```
>>> import torch
>>> 
``` -->
