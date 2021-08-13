# baylib C++ library
<p align="center">
 <img alt="cmake" src="https://img.shields.io/badge/cmake-v3.10.0+-green"/>
 <img alt="c++" src="https://img.shields.io/badge/C++-17 | 20-blue.svg?style=flat&logo=c%2B%2B"/> 
 <img alt="CI build" src="https://github.com/mspronesti/baylib/actions/workflows/ci.yml/badge.svg"/> 
</p>

Baylib is a simple inference engine library for Bayesian networks developed as final project for System Programming class at PoliTO.
The engine supports approximate inference algorithms.

Here's a list of the main requested features:
* Copy-On-Write semantics for the graph data structure, including the conditional probability table (CPT) of each node 
* parallel implementation of the algorithms 
* template-based classes for probability format
* input and output compatible with the [XDSL format](https://support.bayesfusion.com/docs/) provided by the SMILE library
* cmake-based deployment

## Install Dependencies
Under Linux or MacOS, you can use 
the provided script [install_dependencies.sh](scripts/install.sh) as follows
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
cd baylib
mkdir -p build
cd build
cmake ..
make
sudo make install
```
You can now include `baylib` in your projects.
## Usage
Baylib allows to perform approximate inference on Bayesian Networks loaded from xdsl files
or created by hand (either using named nodes or numeric identifiers).

Have a look at [examples](examples) for more.

## External references
* [copy-on-write](https://doc.qt.io/qt-5/qsharedpointer.html)
* [thread pool](https://github.com/bshoshany/thread-pool)
