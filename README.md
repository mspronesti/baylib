# baylib C++ library
<p align="center">
 <img alt="cmake" src="https://img.shields.io/badge/cmake-v3.13+-green"/>
 <img alt="c++" src="https://img.shields.io/badge/C++-17 | 20-blue.svg?style=flat&logo=c%2B%2B"/> 
 <img alt="CI build" src="https://github.com/mspronesti/baylib/actions/workflows/ci.yml/badge.svg"/> 
 <img alt="GPU build" src="https://github.com/mspronesti/baylib/actions/workflows/build-gpu.yml/badge.svg"/>
</p>

Baylib is a simple inference library for discrete Bayesian networks developed as final project for System Programming class at PoliTO.
It supports approximate inference algorithms.

## Main features
Here's a list of the main requested features:
* Copy-On-Write semantics for the graph data structure, including the conditional probability table (CPT) of each node 
* parallel implementation of the algorithms either using C++11 (or higher) threads or GPU computing with [boost compute](https://www.boost.org/doc/libs/1_66_0/libs/compute/doc/html/index.html)
* template-based classes for probability format
* input compatibility with the [XDSL format](https://support.bayesfusion.com/docs/) provided by the SMILE library
* cmake-based deployment

## Currently supported algorithms
* Gibbs Sampling - C++17 threads
* Likelihood Weighting - C++17 threads
* Logic Sampling - GPGPU with boost compute

## Install Dependencies
Under Linux, you can use 
the provided script [install_dependencies.sh](scripts/install_dependencies.sh) as follows
```bash
 cd scripts/
 chmod u+x install_dependencies.sh
./install_dependencies.sh
```

## Install baylib
Under Linux or MacOS, you can 
run the provided script [install.sh](scripts/install.sh) as follows
```bash
cd scripts/
chmod u+x install.sh
sudo ./install.sh
```
alternatively, run the following commands
(assuming you're in the root of the project):
```bash
mkdir build
cd build
cmake ..
make
sudo make install
```
You can now include `baylib` in your projects.

Make sure your `CMakeLists.txt` looks like this
```cmake
find_package(baylib)
# create your executable 
# and whatever you need for
# your project ...
target_link_libraries(<your_executable> baylib)
```
## Usage
Baylib allows performing approximate inference on Bayesian Networks loaded from xdsl files
or created by hand (either using named nodes or numeric identifiers). 

Please notice that the current release
does not support inference when providing evidences.

Have a look at [examples](examples) for more.

## External references
* [copy-on-write](https://doc.qt.io/qt-5/qsharedpointer.html)
* [thread pool](https://github.com/bshoshany/thread-pool)
* [gibbs sampling](http://vision.psych.umn.edu/users/schrater/schrater_lab/courses/AI2/gibbs.pdf)
* [likelihood weighting](https://arxiv.org/pdf/1304.1504.pdf)
* [likelihood weighting pseudo-code](https://github.com/aimacode/aima-pseudocode/blob/master/md/Likelihood-Weighting.md)
* [logic sampling](https://www.academia.edu/35954159/Propagating_Uncertainty_in_Bayesian_Networks_by_Probabilistic_Logic_Sampling)