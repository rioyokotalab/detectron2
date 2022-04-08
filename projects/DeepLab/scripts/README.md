## official BYOL convert to Pytoch

- [github link](https://github.com/chigur/byol-convert)

### prepare install packages

```
$ git clone https://github.com/chigur/byol-convert
$ pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
$ pip install --upgrade "jax[cpu]"
$ pip install dill
$ pip install dm-haiku
```

### download original weights

- [official weights](https://github.com/deepmind/deepmind-research/tree/master/byol#pretraining)

```
$ wget -i byol_weight_links.txt
```

### how to convert
#### convert the weights

```
$ python convert.py pretrain_res50x1.pkl pretrain_res50x1.pth.tar
```

#### validate the weights

```
$ python validate.py pretrain_res50x1.pth.tar /datasets/imagenet/val
```


## official torchvision model to detectron2

- [tool link](https://github.com/rioyokotalab/detectron2/blob/60d7a1fd33cc48e58968659cd3301f3300b2786b/tools/convert-torchvision-to-d2.py)

ex.

```
$ python detectron2/tools/convert-torchvision-to-d2.py pretrain_res50x1.pth.tar pretrain_res50x1.pkl
```

## my flowe checkpoint convert to detectron2

use above tool, but raw file is error.
So, please exec following instrunction

```
$ pushd ~/detectron2/projects/Deeplab/scripts
$ python change2detectron_model_statedict.py --model_path checkpoint0109.pth.tar --out_path checkpoint0109.pth
$ popd 
$ python ~/detectron2/tools/convert-torchvision-to-d2.py checkpoint0109.pth checkpoint0109.pkl
```

