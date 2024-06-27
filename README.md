# Scalable Distributed String Sorting


This is the code to accompany our paper:
_Kurpicz, F., Mehnert, P., Sanders, P., & Schimek, M. (2024). Scalable Distributed String Sorting. arXiv preprint arXiv:2404.16517.

If you use this code in the context of an academic publication, we kindly ask you to [cite it](https://doi.org/10.48550/arXiv.2404.16517):

```bibtex
@article{kurpicz2024scalable,
  title={Scalable Distributed String Sorting},
  author={Kurpicz, Florian and Mehnert, Pascal and Sanders, Peter and Schimek, Matthias},
  journal={arXiv preprint arXiv:2404.16517},
  year={2024}
}
```

## Abstract
String sorting is an important part of tasks such as building index data structures. Unfortunately, current string sorting algorithms do not scale to massively parallel distributed-memory machines since they either have latency (at least) proportional to the number of processors p or communicate the data a large number of times (at least logarithmic). We present practical and efficient algorithms for distributed-memory string sorting that scale to large p. Similar to state-of-the-art sorters for atomic objects, the algorithms have latency of about p1/k when allowing the data to be communicated k times. Experiments indicate good scaling behavior on a wide range of inputs on up to 49152 cores. Overall, we achieve speedups of up to 5 over the current state-of-the-art distributed string sorting algorithms.

## Building

```shell
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Note to run the RQuick variants as standalone algorithms you have to set the CMake option `USE_RQUICK_SORT`.

## Running
For executing the algorithms, run:
```shell
mpiexec -n <NUM_PEs> ./build/distributed_sorter <arguments> # --help provides an overview
```

For reproducing our experiments run
```shell
    cd kaval
    python run-experiments.py <np_supermuc|np_supermuc_rquick|dn_supermuc|dn_supermuc_rquick|real_world_supermuc> --machine <supermuc|generic-job-file> --command-template ./command-templates/supermuc-OpenMPI.txt --search-dirs ../experiment_suites
```

This will be create jobfiles which can then be executed on your system. Note that it is quite likely that you have to adapt the concrete job and command template as these are often cluster-specific. See `python run-experiments.py --help` for further information. 
