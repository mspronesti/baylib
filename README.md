# baylib C++ library
<p align="center">
 <img alt="c++" src="https://img.shields.io/badge/C++-20-blue.svg?style=flat&logo=c%2B%2B"/> 
 <img alt="CI build" src="https://github.com/mspronesti/baylib/actions/workflows/ci.yml/badge.svg"/> 
 <img alt="GPU build" src="https://github.com/mspronesti/baylib/actions/workflows/build-gpu.yml/badge.svg"/>
 <img alt="License" src="https://img.shields.io/github/license/mspronesti/baylib"/>
</p>

Baylib is a parallel inference library for discrete Bayesian networks supporting approximate inference algorithms both in CPU and GPU.

## Main features
Here's a list of the main requested features:
* Copy-On-Write semantics for the graph data structure, including the conditional probability table (CPT) of each node
* parallel implementation of the algorithms either using C++17 threads or GPGPUU optimization
* GPGPU optimization implemented with opencl, using [boost compute](https://www.boost.org/doc/libs/1_66_0/libs/compute/doc/html/index.html) and cuda.
* template-based classes for probability format
* input compatibility with the [XDSL format](https://support.bayesfusion.com/docs/) provided by the SMILE library
* cmake-based deployment

## Currently supported algorithms
* Gibbs Sampling - C++11 threads
* Likelihood Weighting - C++11 threads
* Logic Sampling - GPGPU with boost compute
* Rejection Sampling - C++11 threads
* Adaptive importance sampling - C++11 threads, GPGPU with boost compute

|       algorithm      	         | evidence 	| deterministic nodes 	| multi-threading 	| GPGPU-opencl 	| GPGPU - CUDA |
|:------------------------------:|--------------|-----------------------|-------------------|---------------|--------------|
| gibbs sampling       	         |    &check;   |     *                 |     &check;       |               |              |
| likelihood weighting 	         |    &check;   |     &check;           |     &check;       |               |              |
| logic sampling       	         |    &check;   |     &check;           |                   |  &check;      | &check;      |
| rejection sampling  	         |    &check;   |     &check;           |     &check;       |               |              |
| adaptive importance sampling   |    &check;  	|     &check;           |     &check;       |  &check;      |              |

*It's a very well-known limitation of the Gibbs sampling approach
## Dependencies
* cmake >= 2.8
* boost >= 1.65
* libtbb
* ocl-icd-opencl
* mesa-opencl-icd

Under Linux, you can install the required dependencies using the provided script [install_dependencies.sh](scripts/install_dependencies.sh) as follows
```bash
 cd scripts/
 chmod u+x install_dependencies.sh
./install_dependencies.sh
```

## Install baylib

Using the cmake `FetchContent` directives you can directly setup baylib as follows

```cmake
include(FetchContent)

FetchContent_Declare(
        baylib
        GIT_REPOSITORY https://github.com/mspronesti/baylib.git
)

FetchContent_MakeAvailable(baylib)
# create your executable 
# and whatever you need for
# your project ...
target_link_libraries(<your_executable> baylib)
```
Alternatively
under Linux or MacOS, you can
run the provided script [install.sh](scripts/install.sh) as follows
```bash
cd scripts/
chmod u+x install.sh
sudo ./install.sh
```
another option for the script is running the following commands
(assuming you're in the root of the project):
```bash
mkdir build
cd build
cmake ..
make
sudo make install
```
You can now include `baylib` in your projects.

In the latter two cases, make sure your `CMakeLists.txt` looks like this
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

Have a look at [examples](examples) for more.

## External references
* [copy-on-write](https://doc.qt.io/qt-5/qsharedpointer.html)
* [gibbs sampling](http://vision.psych.umn.edu/users/schrater/schrater_lab/courses/AI2/gibbs.pdf)
* [likelihood weighting](https://arxiv.org/pdf/1304.1504.pdf)
* [likelihood weighting pseudo-code](https://github.com/aimacode/aima-pseudocode/blob/master/md/Likelihood-Weighting.md)
* [logic sampling](https://www.academia.edu/35954159/Propagating_Uncertainty_in_Bayesian_Networks_by_Probabilistic_Logic_Sampling)
* [rejection sampling pseudo-code](https://github.com/aimacode/aima-pseudocode/blob/master/md/Rejection-Sampling.md)
* [adaptive importance sampling](https://arxiv.org/abs/1106.0253)