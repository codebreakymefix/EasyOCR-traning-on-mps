import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
cudnn.benchmark = False
cudnn.deterministic = False
import torch
from pathlib import Path

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt


# print("Torch version:", torch.__version__)
root = Path.cwd()

print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())
# print("Current device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(0))

opt = get_config(root / "config_files/en_filtered_config.yaml")
if __name__ == '__main__':
    train(opt, amp=True)