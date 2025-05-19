# Handmade module for adversarial patch attacking of object detection in Ultralytics framework

Intergrate the code from [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo.git) 

- [cfg](./cfg): folder to place config files
  - attacker.yaml: The config file to build [PatchAttacker](./attacker.py#L15)
  - 30values.txt: values for [NPSCalculator](./loss.py#L116)
    - From [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo.git)