import importlib
import json
import subprocess
from pathlib import Path

import torch
from transformers import T5Config, T5Tokenizer


def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.

    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class

    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def get_git_hash_commit() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def expand_dp_path(path, variables):
    """Expand paths from DeepPavlov configs, uses variables from config's metadata.
    """
    while '{' in path and '}' in path:
        path = path.format(**variables)
    path = Path(path).expanduser()
    return path


def load_experiment(path, t5_configs_path, checkpoint=None, check_commit=True, finetuning_model_config_path=None):
    path = Path(path)
    cfg = json.load((path / 'config.json').open('r'))
    
    finetuning_cfg = json.load((Path(finetuning_model_config_path)).open('r'))
    if (cfg['model_cfg'] != finetuning_cfg['t5_config']) and (cfg['model_cls'] == finetuning_cfg['model_cls']):
        print(f"\n\nmodel_cls match, model_cfg != t5_config, loading finetuning t5_config\n\n")
        model_cfg = Path(t5_configs_path) / finetuning_cfg['t5_config'] if finetuning_cfg['t5_config'] is not None else None
    else:    
        model_cfg = Path(t5_configs_path) / cfg['model_cfg'] if cfg['model_cfg'] is not None else None
    
    model_cls = get_cls_by_name(cfg['model_cls'])
    if check_commit:
        assert cfg['COMMIT'] == get_git_hash_commit(), f"expected commit {cfg['COMMIT']}, " \
                                                       f"but current is {get_git_hash_commit()}"
    # take latest checkpoint
    if checkpoint is None:
        checkpoint = list(sorted(path.glob('*.pth'), key=lambda x: x.stat().st_ctime))[-1]

    if model_cfg is None:
        t5config = T5Config.from_pretrained(cfg['base_model'])
    else:
        t5config = T5Config.from_json_file(model_cfg)

    t5tokenizer = T5Tokenizer.from_pretrained(cfg['base_model'])

    model = model_cls(config=t5config)

    state_dict = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(state_dict["model_state_dict"])
    print(f'Model was loaded from: {checkpoint}')
    model.eval()
    return model, t5tokenizer

def load_finetuning_model(pretraining_model, path, finetuning_model_config_path, checkpoint):
    
    cfg = json.load((Path(finetuning_model_config_path)).open('r'))
    
    path = Path(path)
    cfg_raw = json.load((path / 'config.json').open('r'))
    print(f'\n\n{path} \n {cfg_raw}')
    
    if cfg_raw['model_cls'] == "modeling_t5:T5WMForConditionalGeneration":
        print('\n\nLoaded model is already with working memory, skipping copying weights')
        return pretraining_model
    else:
        model_cls = get_cls_by_name(cfg['model_cls'])
        t5_config = T5Config.from_json_file(Path(cfg['t5_config']))
        model = model_cls(config=t5_config)


        state_dict = torch.load(str(checkpoint), map_location='cpu')
        pretrained_dict = state_dict["model_state_dict"]
        model_dict = model.state_dict()

        '''
        with open("./wm_state_dict.txt", "w") as f:
            for k in model_dict.keys():
                f.write(f"{str(k)}, {model_dict[k].size()}\n")
        with open("./pretrained_state_dict.txt", "w") as f:
            for k in pretrained_dict.keys(): 
                f.write(f"{str(k)}, {pretrained_dict[k].size()}\n")
        print(f"model_dict = {model.state_dict().keys()}")
        '''

        pretrained_dict['decoder.wm_tok_type_embed_tokens.weight'] = model_dict['decoder.wm_tok_type_embed_tokens.weight']
        pretrained_dict['wm_tok_type_embedding.weight'] = model_dict['wm_tok_type_embedding.weight']


        pretrained_dict['lm_head.weight'] = torch.cat([pretrained_dict['lm_head.weight'], 
                                                       model_dict['wm_tok_type_embedding.weight']],dim=0)

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(pretrained_dict)


        print(f'Model was loaded from: {checkpoint}')
        model.eval()

        return model
