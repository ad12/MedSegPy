# Use Configs

In MedSegPy's config structure, config types are 1-to-1 with model architectures.
This means that different architectures have different config types. For a list
of config types, see [modules/config][../modules/config]. The base `Config`
class is abstract; use the config type corresponding to the architecture
you would like to build.

Configs support ini/yaml files and uses [config parser](https://docs.python.org/2/library/configparser.html) for reading and writing configs.

### Use Configs

Some basic usage of the `Config` object is shown below. We use the 2D U-Net
config for clarity.:
```python
from medsegpy.config import UNetConfig
cfg = UNetConfig()    # obtain medsegpy's default config for 2d U-Net
cfg.xxx = yyy      # set values for data
cfg.merge_from_file("my_cfg.yaml")   # load values from a file
cfg.merge_from_file("my_cfg.ini")  # supports ini files too

cfg.merge_from_list(["USE_EARLY_STOPPING", True])  # can also load values from a list of str
cfg.summary()  # print a formatted summary
```

To see a list of example configs in medsegpy, see [configs](../../configs).


### Best Practice with Configs

1. Keep the configs you write simple: don't include keys that do not affect the experimental setting.

2. Keep a version number in your configs (or the base config), e.g., `VERSION: 2`,
   for backward compatibility.
	 We print a warning when reading a config without version number.
   The official configs do not include version number because they are meant to
   be always up-to-date.

3. Save a full config together with a trained model, and use it to run inference.
   This is more robust to changes that may happen to the config definition
   (e.g., if a default value changed), although we will try to avoid such changes.