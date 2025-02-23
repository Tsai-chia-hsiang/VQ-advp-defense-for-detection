import yaml
import json
from argparse import Namespace

def load_yaml(yf):
    with open(yf, "r") as fp:
        c = yaml.safe_load(fp)
    return c

def write_yaml(o, f):
    with open(f, "w+") as yf:
        yaml.dump(o, yf, default_flow_style=False, allow_unicode=True)


def read_json(f, encoding='utf-8'):
    with open(f, "r", encoding=encoding) as jf:
        return json.load(jf)

def write_json(o, f):
    with open(f, "w+") as jf:
        json.dump(o, jf, indent=4, ensure_ascii=False)

def args2dict(args:Namespace|dict, wanted_keys:list[str]=None, neg_prefix:tuple[str]=('no_', 'not_')) -> dict:
    
    def to_postive(k:str, v:bool)->tuple[str, bool]:
        for n in neg_prefix:
            if k.startswith(n):
                return k.replace(n, ''), not v
        return k, v
    
    value = vars(args) if not isinstance(args, dict) else args
    processed_value = {}

    if wanted_keys is None:
        wanted_keys = value
    
    for k in wanted_keys:
        pk = k
        v = value[k]
        if isinstance(value[k], bool):
            pk, v = to_postive(k=pk, v=v)
        
        processed_value[pk] = v

    return processed_value


