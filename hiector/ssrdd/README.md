# Single-Stage Rotation-Decoupled Detector for Oriented Object

This code is taken and adapted from [this GitHub repository](https://github.com/Capino512/pytorch-rotation-decoupled-detector), which implements the algorithm reported in the paper **Single-Stage Rotation-Decoupled Detector for Oriented Object**. [[Paper]](https://www.mdpi.com/2072-4292/12/19/3262/htm) [[PDF]](https://www.mdpi.com/2072-4292/12/19/3262/pdf)

## Compile

```bash
# 'rbbox_batched_nms' will be used as post-processing in the interface stage
# use gpu, for Linux only
cd $PATH_ROOT/utils/box/ext/rbbox_overlap_gpu
python setup.py build_ext --inplace

# alternative, use cpu, for Windows and Linux
cd $PATH_ROOT/utils/box/ext/rbbox_overlap_cpu
python setup.py build_ext --inplace
```

## Citation

```
@article{rdd,
    title={Single-Stage Rotation-Decoupled Detector for Oriented Object},
    author={Zhong, Bo and Ao, Kai},
    journal={Remote Sensing},
    year={2020}
}
```

## TODO

things to change/parameterize:

 * [ ] have a single script for training/evaluation, e.g. `execute.py` where a flag is specified for either training/testing
 * [ ] have a .json config file with model parameters
 * [ ] extract the `stride` parameter of the backbone architecture (i.e. head), to allow to change this as hyper-param (used in SPOT/S2)
 * [ ] test/remove/replace the nms functionality provided in `utils/box/ext`
 * [ ] test/remove parallel implementation of training