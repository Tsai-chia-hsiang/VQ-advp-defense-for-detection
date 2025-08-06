from ultralytics_advpattack_lib.dummuy_validation import yolo_val
from pathlib import Path


def main():
    proj = Path("exps/advyolo_obj/cdfull")
    data = Path("datasets/INRIAPerson/inria.yaml")
    r = yolo_val(proj=proj, default_data=data)
    print(r)


if __name__ == "__main__":
    main()
