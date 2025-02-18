import yaml
import json

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

