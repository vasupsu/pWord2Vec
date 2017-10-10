# pSGNScc
This is the C++ implementation of the Context Combining optimization of Word2Vec described in the paper titled,

### Optimizing Word2Vec Performance on Multicore Systems, accepted at IA^3 2017 - the Seventh Workshop on Irregular Applications: Architectures & Algorithms, co-located with SC17 

The code is developed based on the [original pWord2Vec](https://github.com/IntelLabs/pWord2Vec.git) implementation described in the paper [Parallelizing Word2Vec in Shared and Distributed Memory](https://arxiv.org/abs/1604.04661), arXiv, 2016.

## License
All source code files in the package are under [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Prerequisites
The code is developed and tested on UNIX-based systems with the following software dependencies:

- [Intel Compiler](https://software.intel.com/en-us/qualify-for-free-software) (The code is optimized on Intel CPUs)
- OpenMP (No separated installation is needed once Intel compiler is installed)
- MKL (The latest version "16.0.0 or higher" is preferred as it has been improved significantly in recent years)
- [HyperWords](https://bitbucket.org/omerlevy/hyperwords) (for model accuracy evaluation)
- Numactl package (for multi-socket NUMA systems)

## Environment Setup
* Install Intel C++ development environment (i.e., Intel compiler, OpenMP, MKL "16.0.0 or higher". [free copies](https://software.intel.com/en-us/qualify-for-free-software) are available for some users)
* Enable Intel C++ development environment
```
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64 (please point to the path of your installation)
source /opt/intel/impi/latest/compilers_and_libraries/linux/bin/compilervars.sh intel64 (please point to the path of your installation)
```
* Install numactl package
```
sudo yum install numactl (on RedHat/Centos)
sudo apt-get install numactl (on Ubuntu)
```

## Quick Start
1. Download the code: ```git clone git@github.com:vasupsu/pWord2Vec.git```
2. Run make to build the package  
This installation will  produce three binaries: word2vec, pWord2Vec and pSGNScc. These correspond to the original implementation of Word2Vec found in this [GIT repository](https://github.com/dav/word2vec.git), [original pWord2Vec](https://github.com/IntelLabs/pWord2Vec.git) and our pSGNScc context combining approach. The other implementations are included for performance comparison and verification.
3. Download the data: ```cd data; .\getText8.sh or .\getBillion.sh```
4. The directory IA3_AE_test_cases contain BASH test scripts for validating the results in our IA^3 submission. Each test script validates one Figure or Table present in the Experiments and Results section of the paper. The name of each test script corresponds to the Figure or Table number in the paper it validates.

## Reference
1. [Optimizing Word2Vec Performance on Multicore Systems](https://github.com/vasupsu/pWord2Vec), accepted at IA^3 2017.
2. [Parallelizing Word2Vec in Shared and Distributed Memory](https://arxiv.org/abs/1604.04661), arXiv, 2016.
3. [Parallelizing Word2Vec in Multi-Core and Many-Core Architectures](https://arxiv.org/abs/1611.06172), in NIPS workshop on Efficient Methods for Deep Neural Networks, Dec. 2016.

For questions, please contact us at vxr162@psu.edu
