# FlatCTR

FlatCTR is a high-performance toolkit to train Logistic Regression(LR) or Factorization Machines(FM) on large-scale sparse dataset.

## Building

```shell
git clone --recursive https://github.com/maxuewei2/FlatCTR.git
cd FlatCTR
mkdir build && cd build
cmake ..
make
```


## Usage
1. Run `./fastctr -h` to display a list of all supported options.
2. Run `./flatctr` to train on the sample dataset.
3. For online training, use the `-i` option to load a trained model:
   `./flatctr -i ../output/model.txt`

### Data Format
The input data should be in the libsvm format.