import logging
import numpy as np
import os

from keras import Input
from collections import defaultdict
from medsegpy import config
from medsegpy.modeling.meta_arch.build import META_ARCH_REGISTRY
from medsegpy.modeling import Model, get_model, model_from_json
from medsegpy.modeling.meta_arch import build_model
from medsegpy.modeling.layers.convolutional import ConvStandardized2D
from medsegpy.modeling.layers.normalization import GroupNormalization
from medsegpy.utils import dl_utils
from typing import Sequence, Tuple


logger = logging.getLogger(__name__)


def get_all_layers(model: Model) -> Sequence[Tuple]:
    name_to_layer = []
    for layer in model.layers:
        if "input" not in layer.name:
            if "model" in layer.name:
                name_to_layer.extend(get_all_layers(layer))
            else:
                name_to_layer.append((layer.name, layer))
    return name_to_layer


def word_contains_words(word: str,
                        include_words: Sequence[str],
                        exclude_words: Sequence[str]) -> bool:
    for iw in include_words:
        if iw not in word:
            return False
    for ew in exclude_words:
        if ew in word:
            return False
    return True


def remove_keys_from_config(config: dict):
    keys = ["name",
            "trainable",
            "kernel_initializer",
            "bias_initializer"]
    for key in keys:
        _ = config.pop(key, None)


def get_layers_to_load(model: Model,
                       cfg: config.Config) \
        -> Tuple[Sequence, Sequence]:
    # Get instance of config used to train the pretrained model
    assert type(cfg.PRETRAINED_WEIGHTS_PATH) is dict, \
        "'PRETRAINED_WEIGHTS_PATH' must a dictionary"
    assert os.path.exists(cfg.PRETRAINED_CONFIG_PATH), \
        f"'PRETRAINED_CONFIG_PATH' does not exist! " \
        f"Path: {cfg.PRETRAINED_CONFIG_PATH}"
    pt_info = cfg.PRETRAINED_WEIGHTS_PATH
    pt_config_path = cfg.PRETRAINED_CONFIG_PATH
    pt_model_name = config.get_model_name(pt_config_path)
    pt_config = config.get_config(pt_model_name, create_dirs=False)
    pt_config.merge_from_file(pt_config_path)

    # Get path of weights to load
    weight_file = pt_info["path"]
    if not weight_file:
        # If weight_file not given, choose best weights based on
        # performance on validation set
        weight_file = dl_utils.get_weights(pt_config.OUTPUT_DIR)
    logger.info(f"Loading specific weights from {weight_file}")

    pt_model = None
    pt_model_json_file = os.path.join(pt_config.OUTPUT_DIR, "model.json")
    # Try loading pretrained model from JSON file, if this training run
    # is not the frozen run of self-supervised learning
    if os.path.isfile(pt_model_json_file):
        try:
            with open(pt_model_json_file) as f:
                json_str = f.read()
            pt_model = model_from_json(
                json_str,
                custom_objects={
                    "Model": Model,
                    "ConvStandardized2D": ConvStandardized2D,
                    "GroupNormalization": GroupNormalization
                }
            )
        except Exception as e:
            print(e)

    if pt_model is None:
        # Build pretrained model from config
        pt_builder = META_ARCH_REGISTRY.get(pt_model_name)(pt_config)
        pt_model = pt_builder.build_model()
        logger.info("Built pretrained model from config")
    else:
        logger.info("Built pretrained model from JSON")

    # Load weights into pretrained model
    pt_model.load_weights(weight_file)

    # Find all layers of both pretrained and current model
    pt_layer_list = get_all_layers(pt_model)
    cur_layer_list = get_all_layers(model)
    pt_idx_layers = defaultdict(list)
    cur_idx_layers = defaultdict(list)
    weights_to_load = pt_info["weights"]

    # Determine list of layers to load
    assert type(weights_to_load) is list, \
        "Key 'weights' of PRETRAINED_WEIGHTS_PATH must be a list"
    pt_layer_name_to_idx = {}
    cur_layer_name_to_idx = {}
    total_pt_load_layers = []
    total_cur_load_layers = []
    for weight_idx, weight_info in enumerate(weights_to_load):
        assert "include_words" in weight_info, \
            f"Key 'include_word' is required in " \
            f"weight dictionary at index {weight_idx}"
        assert "exclude_words" in weight_info, \
            f"Key 'exclude_word' is required in " \
            f"weight dictionary at index {weight_idx}"
        assert "slice_indices" in weight_info, \
            f"Key 'slice_indices' is required in " \
            f"weight dictionary at index {weight_idx}"

        # Check format of range of weight indices for each weight key
        range_list = weight_info["slice_indices"]
        assert len(range_list) == 2, \
            f"The length of the list for key 'slice_indices' must " \
            f"2. Error occurs for weight dictionary at index {weight_idx}."
        assert isinstance(range_list[0], int), \
            f"The first value in the list for key 'slice_indices' must " \
            f"be an integer. Error occurs for weight dictionary at " \
            f"index {weight_idx}."

        for pt_layer_tuple in pt_layer_list:
            layer_name = pt_layer_tuple[0]
            if word_contains_words(layer_name,
                                   weight_info["include_words"],
                                   weight_info["exclude_words"]):
                assert layer_name not in pt_layer_name_to_idx, \
                    f"Layer {layer_name} of pretrained model is already " \
                    f"assigned to weight dictionary at index " \
                    f"{pt_layer_name_to_idx[layer_name]}"
                pt_idx_layers[weight_idx].append(pt_layer_tuple)
                pt_layer_name_to_idx[layer_name] = weight_idx
        assert weight_idx in pt_idx_layers, \
            f"No layers found for weight dictionary at index {weight_idx}"
        for cur_layer_tuple in cur_layer_list:
            layer_name = cur_layer_tuple[0]
            if word_contains_words(layer_name,
                                   weight_info["include_words"],
                                   weight_info["exclude_words"]):
                assert layer_name not in cur_layer_name_to_idx, \
                    f"Layer {layer_name} of current model is already " \
                    f"assigned to weight dictionary at index " \
                    f"{cur_layer_name_to_idx[layer_name]}"
                cur_idx_layers[weight_idx].append(cur_layer_tuple)
                cur_layer_name_to_idx[layer_name] = weight_idx
                if range_list[1] != "until":
                    matched_pt_layer_tuple = pt_idx_layers[weight_idx][
                        len(cur_idx_layers[weight_idx]) - 1
                        ]
                    matched_pt_config = matched_pt_layer_tuple[1].get_config()
                    cur_config = cur_layer_tuple[1].get_config()
                    remove_keys_from_config(matched_pt_config)
                    remove_keys_from_config(cur_config)
                    assert matched_pt_config == cur_config, \
                        f"Config of layer in current model with name " \
                        f"{layer_name} does not match config of corresponding " \
                        f"layer in pretrained model with name " \
                        f"{matched_pt_layer_tuple[0]}"
        assert weight_idx in cur_idx_layers, \
            f"No layers found for weight dictionary at index {weight_idx}"

        # Check that both the current model and the pretrained model have
        # the same number of layers in each key group
        num_pt_layers = len(pt_idx_layers[weight_idx])
        num_cur_layers = len(cur_idx_layers[weight_idx])
        assert num_pt_layers == num_cur_layers, \
            f"Pretrained model and current model have different numbers " \
            f"of layers for weight dictionary at index {weight_idx}. " \
            f"Pretrained model has {num_pt_layers} and current model " \
            f"has {num_cur_layers}."

        if range_list[1] == "until":
            pt_load_layers = []
            cur_load_layers = []
            until_cur_layer_tuple = cur_idx_layers[weight_idx][0]
            for layer_idx in range(len(cur_layer_list)):
                cur_layer_tuple = cur_layer_list[layer_idx]
                pt_layer_tuple = pt_layer_list[layer_idx]
                pt_layer_name = pt_layer_tuple[0]
                cur_layer_name = cur_layer_tuple[0]

                # Check if current layer is the same as pretrained layer
                if not cur_layer_name == until_cur_layer_tuple[0]:
                    cur_config = cur_layer_tuple[1].get_config()
                    matched_pt_config = pt_layer_tuple[1].get_config()
                    remove_keys_from_config(cur_config)
                    remove_keys_from_config(matched_pt_config)
                    assert cur_config == matched_pt_config, \
                        f"Config of layer in current model with name " \
                        f"{cur_layer_tuple[0]} does not match config of " \
                        f"corresponding layer in pretrained model with name " \
                        f"{pt_layer_tuple[0]}"
                    assert pt_layer_name not in pt_layer_name_to_idx, \
                        f"Layer {pt_layer_name} of pretrained model is already " \
                        f"assigned to weight dictionary at index " \
                        f"{pt_layer_name_to_idx[pt_layer_name]}"
                    assert cur_layer_name not in cur_layer_name_to_idx, \
                        f"Layer {cur_layer_name} of current model is already " \
                        f"assigned to weight dictionary at index " \
                        f"{cur_layer_name_to_idx[cur_layer_name]}"
                    pt_load_layers.append(pt_layer_tuple)
                    pt_layer_name_to_idx[pt_layer_name] = weight_idx
                    cur_load_layers.append(cur_layer_tuple)
                    cur_layer_name_to_idx[cur_layer_name] = weight_idx
                else:
                    break
        else:
            if range_list[1] == 'None':
                range_list[1] = None
            assert range_list[1] is None or isinstance(range_list[1], int), \
                f"The second value in the list for key 'slice_indices' must " \
                f"be either None or an integer. Error occurs for weight " \
                f"dictionary at index {weight_idx}."

            pt_load_layers = pt_idx_layers[weight_idx][
                             range_list[0]: range_list[1]
                             ]
            cur_load_layers = cur_idx_layers[weight_idx][
                              range_list[0]: range_list[1]
                              ]
        total_pt_load_layers.extend(pt_load_layers)
        total_cur_load_layers.extend(cur_load_layers)

    assert len(total_pt_load_layers) == len(total_cur_load_layers), \
        f"Must have same number of layers in pretrain layer list and " \
        f"current layer list. Found {len(total_pt_load_layers)} pretrain " \
        f"layers and {len(total_cur_load_layers)} current layers."

    return total_pt_load_layers, total_cur_load_layers


def load_specific_weights(model: Model,
                          cfg: config.Config,
                          debug: bool = True):
    pt_load_layers, cur_load_layers = get_layers_to_load(model, cfg)
    num_layers_loaded = 0
    loaded_layer_names = []
    for cur_layer_tuple, pt_layer_tuple in zip(
            cur_load_layers, pt_load_layers):
        cur_layer_name = cur_layer_tuple[0]
        pt_layer = pt_layer_tuple[1]
        cur_layer = cur_layer_tuple[1]
        layer_weight = pt_layer.get_weights()
        if layer_weight:
            prev_weight = cur_layer.get_weights()
            cur_layer.set_weights(layer_weight)
            new_weight = cur_layer.get_weights()
            assert np.any(
                new_weight[0] != prev_weight[0]
            ), \
                f"Loading weights for layer {cur_layer_name} " \
                f"in current model did not work: Weights are still " \
                f"random!"
            assert np.all(
                new_weight[0] == layer_weight[0]
            ), \
                f"Loading weights for layer {cur_layer_name} " \
                f"{cur_layer_name} did not work: New weights do" \
                f"not match pretrained weights"
            num_layers_loaded += 1
            loaded_layer_names.append(cur_layer_name)
            if cfg.FREEZE_PRETRAINED:
                cur_layer.trainable = False
            else:
                cur_layer.trainable = True

    # Check that the right number of layers were loaded
    assert num_layers_loaded > 0, \
        "Loading did not work: No weights were loaded!"

    if debug:
        pt_load_layers, cur_load_layers = get_layers_to_load(model, cfg)
        for cur_layer_tuple, pt_layer_tuple in zip(
                cur_load_layers, pt_load_layers):
            cur_layer_name = cur_layer_tuple[0]
            pt_layer = pt_layer_tuple[1]
            cur_layer = cur_layer_tuple[1]
            loaded_weight = pt_layer.get_weights()
            if loaded_weight:
                cur_weight = cur_layer.get_weights()
                assert np.all(
                    loaded_weight[0] == cur_weight[0]
                ), \
                    f"Loading weights for layer {cur_layer_name} in " \
                    f"current model did not work: Model is not " \
                    f"modified!"
                if cfg.FREEZE_PRETRAINED:
                    assert not cur_layer.trainable, \
                        f"Layer '{cur_layer_name}' is not frozen!"
                else:
                    assert cur_layer.trainable, \
                        f"Layer '{cur_layer_name}' is frozen!"

    logger.info("All weights were successfully loaded!")
    logger.info(f"Loaded layer names in current model: {loaded_layer_names}")


class SelfSupervisedInfo(object):

    _LOADED_LAYER_NAMES = []

    @staticmethod
    def init_self_supervised(cfg: config.Config):
        # Check config format for self-supervised learning
        if cfg.LEARNING_TAG == "self-supervised":
            model = SelfSupervisedInfo.temp_build_model(cfg)
            if cfg.SS_LOADED_LAYERS:
                SelfSupervisedInfo._LOADED_LAYER_NAMES = cfg.SS_LOADED_LAYERS.copy()
            else:
                assert type(cfg.PRETRAINED_WEIGHTS_PATH) is dict, \
                    "'PRETRAINED_WEIGHTS_PATH' must a dictionary"
                assert os.path.exists(cfg.PRETRAINED_CONFIG_PATH), \
                    f"'PRETRAINED_CONFIG_PATH' does not exist! " \
                    f"Path: {cfg.PRETRAINED_CONFIG_PATH}"

                # Get info of weights to be loaded
                pt_info = cfg.PRETRAINED_WEIGHTS_PATH
                weights_to_load = pt_info["weights"]

                # Ensure weights_to_load is of the right format for
                # self-supervised learning
                assert type(weights_to_load) is list, \
                    "Key 'weights' of PRETRAINED_WEIGHTS_PATH must be a list"
                assert len(weights_to_load) == 1, \
                    "Must have only one weight dictionary defined in " \
                    "PRETRAINED_WEIGHTS_PATH"
                weight_info = weights_to_load[0]
                assert "slice_indices" in weight_info, \
                    "Key 'slice_indices' is required in weight dictionary"
                range_list = weight_info["slice_indices"]
                assert len(range_list) == 2, \
                    "The length of the list for key 'slice_indices' must 2"
                '''
                assert range_list[1] == "until", \
                    "The last index of 'slice_indices' in weight dictionary " \
                    "must be the word 'until' for self-supervised learning"
                '''

                # Get layers to be loaded
                _, total_cur_load_layers = get_layers_to_load(model, cfg)

                # Initialize _LOADED_LAYERS_NAMES
                for layer_tuple in total_cur_load_layers:
                    SelfSupervisedInfo._LOADED_LAYER_NAMES.append(
                        layer_tuple[0]
                    )
            logger.info(f"SelfSupervisedInfo._LOADED_LAYER_NAMES: "
                        f"{SelfSupervisedInfo._LOADED_LAYER_NAMES}")

    @staticmethod
    def clear():
        SelfSupervisedInfo._LOADED_LAYER_NAMES.clear()

    @staticmethod
    def get_loaded_layer_names():
        return SelfSupervisedInfo._LOADED_LAYER_NAMES.copy()

    @staticmethod
    def temp_build_model(cfg):
        try:
            return build_model(cfg)
        except KeyError:
            return get_model(cfg)

    @staticmethod
    def set_to_inference(layer_name: str):
        if layer_name in SelfSupervisedInfo._LOADED_LAYER_NAMES:
            return {"training": False}
        else:
            return {}

    @staticmethod
    def create_self_supervised_model(model: Model,
                                     cfg: config.Config) -> Model:
        # Get instance of config used to train the pretrained model
        pt_info = cfg.PRETRAINED_WEIGHTS_PATH

        # Find all layers of current model
        cur_layer_list = get_all_layers(model)

        # Get weight info
        weights_to_load = pt_info["weights"]

        # Ensure weights_to_load is of the right format for
        # self-supervised learning
        assert type(weights_to_load) is list, \
            "Key 'weights' of PRETRAINED_WEIGHTS_PATH must be a list"
        assert len(weights_to_load) == 1, \
            "Must have only one weight dictionary defined in " \
            "PRETRAINED_WEIGHTS_PATH"
        weight_info = weights_to_load[0]
        assert "slice_indices" in weight_info, \
            "Key 'slice_indices' is required in weight dictionary"
        range_list = weight_info["slice_indices"]
        assert len(range_list) == 2, \
            "The length of the list for key 'slice_indices' must 2"
        assert range_list[1] == "until", \
            "The last index of 'slice_indices' in weight dictionary " \
            "must be the word 'until' for self-supervised learning"

        cur_layer_name = ""
        found_name = False
        for layer_idx in range(len(cur_layer_list)):
            cur_layer_tuple = cur_layer_list[layer_idx]
            cur_layer_name = cur_layer_tuple[0]
            if word_contains_words(cur_layer_name,
                                   weight_info["include_words"],
                                   weight_info["exclude_words"]):
                found_name = True
                break
        assert found_name, \
            "No layers found in current model that match description in " \
            "weight dictionary"

        # Split model
        split_layer = model.get_layer(name=cur_layer_name)
        if type(model.input) is list:
            pt_model_inputs = [Input(sub_input.shape) for sub_input in model.input]
        else:
            pt_model_inputs = Input(model.input.shape)
        pretrain_model = Model(inputs=pt_model_inputs,
                               outputs=split_layer.input,
                               name="pretrain_model")
        if type(split_layer.input) is list:
            model_2_inputs = [Input(sub_input.shape) for sub_input in split_layer.input]
        else:
            model_2_inputs = Input(split_layer.input.shape)
        remainder_model = Model(inputs=model_2_inputs,
                                outputs=model.output,
                                name="remainder_model")
        # Define new model
        pretrain_model.trainable = not cfg.FREEZE_PRETRAINED
        x = pretrain_model(model.input, training=False)
        x = remainder_model(x)
        new_model = Model(inputs=model.input, outputs=x)

        return new_model

    @staticmethod
    def check_self_supervised_loading(pretrained_portion: Model,
                                      cfg: config.Config):
        """Checks that the weights of all the layers of the pretrained
        portion of the model will be loaded into the current model.
        """
        last_pt_layer_name = pretrained_portion.layers[-1].name

        # Get info of weights to be loaded
        pt_info = cfg.PRETRAINED_WEIGHTS_PATH
        weights_to_load = pt_info["weights"]
        weight_info = weights_to_load[0]

        # Find all layers of pretrained portion of the model
        pt_layer_list = get_all_layers(pretrained_portion)

        # This will work because the pretrained portion is part
        # of the current model
        prev_layer_name = ""
        for layer_idx in range(len(pt_layer_list)):
            cur_layer_tuple = pt_layer_list[layer_idx]
            cur_layer_name = cur_layer_tuple[0]
            if word_contains_words(cur_layer_name,
                                   weight_info["include_words"],
                                   weight_info["exclude_words"]):
                break
            prev_layer_name = cur_layer_name

        assert prev_layer_name == last_pt_layer_name, \
            f"Last layer loaded (name = {prev_layer_name}) does match last layer " \
            f"of pretrained portion of current model " \
            f"(name = {last_pt_layer_name})."
