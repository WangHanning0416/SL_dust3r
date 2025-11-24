import os
from omegaconf import OmegaConf
from easydict import EasyDict
import megfile
import numbers
import numpy as np
from collections.abc import Iterable

__DATA_TYPE = ['float', 'int']  # 默认是float.

def __load_yaml_leaf(path):
    with megfile.smart_open(path, 'r') as f:
        cfg = OmegaConf.load(f)
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg)
    return cfg

def __recursive_load_yaml(cfg:dict):
    update = {}
    for k, v in cfg.items():
        if isinstance(v, str) and v.endswith(".yaml") and megfile.smart_isfile(v):
            v = __load_yaml_leaf(v)
            v = __recursive_load_yaml(v)
        elif hasattr(v, 'items'):
            v = __recursive_load_yaml(v)
        update[k] = v
    return EasyDict(update)

def __override_model_kwargs(cfg:dict):
    if 'model' in cfg and "override_kwargs" in cfg['model']:
        for k, v in cfg['model']["override_kwargs"].items():
            cfg['model']['kwargs'][k] = v
    return cfg


def load_config(path):
    '''recursively load all .yaml files.'''
    with megfile.smart_open(path, 'r') as f:
        cfg = OmegaConf.load(f)
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg)
    cfg = __recursive_load_yaml(cfg)
    cfg = __override_model_kwargs(cfg)
    return cfg


def args_parse_bool_str(args):
    for k, v in vars(args).items():
        if not isinstance(v, str):
            continue
        if v.lower() == 'true':
            setattr(args, k, True)
        elif v.lower() == 'false':
            setattr(args, k, False)
    return args


def random_sample_properties_from_config(config, key=None, rng=None):
    '''
    如果传进来的leaf是区间，就返回在区间内的均匀随机数；如果leaf是str or number, 就直接返回  
    也就是说，这个函数可以接受randomize_config，也可以接受已经随即完的配置文件，原封不动返回去.  
    注意只有标量或长度为3的向量，长度为2会被认为是个区间.  
    区间都是闭区间.
    '''
    if hasattr(config, 'items'):   # 本身是个字典
        return {k: random_sample_properties_from_config(v, k, rng) for k, v in config.items()}
    else:  # 本身是leaf, 需要执行sample.
        return __random_sample_properties_from_leaf(config, key, rng)
        
def __random_sample_properties_from_leaf(leaf, key=None, rng=None):
    if leaf is None or isinstance(leaf, (str, numbers.Number)) or leaf == []:
        return leaf
    else:
        if isinstance(leaf[0], str) and leaf[0] in __DATA_TYPE and len(leaf) == 3:  # ['int'/'float', left, right]
            dtype = leaf[0]
            _interval = leaf[1:]
        elif isinstance(leaf[0], numbers.Number) and len(leaf) == 2:  # 长度为2会被认为是float区间！
            dtype = 'float'
            _interval = leaf
            # r = np.random.rand()
            # return r * (leaf[1] - leaf[0]) + leaf[0]
        else:  # leaf什么都不是
            if isinstance(leaf, Iterable):
                return [__random_sample_properties_from_leaf(v, None, rng) for v in leaf]
            else:
                return leaf
            
        # sample
        if dtype == 'float':
            r = np.random.rand() if rng is None else rng.random()
            return r * (leaf[1] - leaf[0]) + leaf[0]
        elif dtype == 'int':
            return np.random.randint(int(_interval[0]), int(_interval[1] + 1)) if rng is None else rng.integers(int(_interval[0]), int(_interval[1]) + 1)
        else:
            raise NotImplementedError