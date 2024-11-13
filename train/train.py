import torch
import torchaudio
import field
import parser
import argparse
import whisper
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer
from load_dataset import load_multiple_datasets, token
import evaluate

from dataclasses import dataclass, field
from model.MSTT_1D_v4 import SpeechToTextModel
import evaluate


config_parser = parser = argparse.ArgumentParser(description="Train Config",add_help=False)

parser.add_argument('-c','--config',default='',type=str,metavar='FILE',help='YAML congih file specifying default arguments')

parser = argparse.ArgumentParser(description='Pytorch Multilingual Speech dataset training')

group = parser.add_argument_group('Dataset Parameters')

# Dataset Parameters
parser.add_argument('data dir',metavar='DIR',help='path to dataset')
group.add_argument('--dataset','-d',metavar='NAME',default='',help='')
group.add_argument('--train-split',metavar='NAME',default='train')

# Model Parameters

group = parser.add_argument_group('Model Parameters')

group.add_argument('--d_model',default=512,type=int,metavar='DIMENSION',
                   help='Dimension of the model')
group.add_argument('--teacher_mode',default='base',type=str,metavar='ENCODER',
                   help='Type of encoder')
group.add_argument('--k_dim',default=512,type=int,metavar='K-DIMENSION',
                   help='Linformer k,v projection dimension')
group.add_argument('--depth',default=3,type=int,metavar='DEPTH',
                   help='Number of decoder layer')
group.add_argument('--num_heads',default=8,type=int,metavar='HEADS',
                   help='Number of Transformer heads')
group.add_argument('--mamba_version',default='v3',type=str,metavar='MAMBA-VERSION',
                   help='Bi-directional Mamba version = v1,v2(mamba vision),v3(our)')
group.add_argument('--learnable_params',default=True,type=bool,metavar='DIFFERENTIAL-TRANSFORMER',
                   help='Differential Transformer Method in cross-attention')
group.add_argument('--flash_attn',default=True,type=bool,metavar='FLASH-ATTENTION',
                   help='In transformer-based cross attention using flash attention')
group.add_argument('--mode',default='Train',type=str,metavar='MODE',
                   help='For Decoder positional encoding choose mode = Train, Inference')
group.add_argument('--linformer',default=True,type=bool,metavar='LINFORMER',
                   help='choose using Linformer or not')   
group.add_argument('--share_kv',default=False,type=bool,metavar='SHARE-KV',
                   help='using k,v weight or only k weight')

# Optimizer Parameters

group = parser.add_argument_group('Optimizer parameters')

group.add_argument('--opt',default='adamW',type=str,metavar='OPTIMIZER',
                   help='Optimizer (defatul: AdamW)')
group.add_argument('--weight decay',default=0.0,type=float,
                   help='weight decay (default: 0.0)')

# Scheduler parameters

group = parser.add_argument_group('Scheduler parameters')
group.add_argument('')

class ModelArguments:
    """
    Model hyper-parameters
    
    """
    d_model: int = field(
        metadata={"help" : 'Model Dimension according to teacher encoder'}
    )
    teacher_mode: str = field(
        metadata={'help':'whisepr audio feature extractor mode = tiny, tiny.en, base, base.en, small, small, medium, large, large_v2'}
    )
    k_dim: int = field(
        metadata={'help': 'Linformer k diemnsion'}
    )
    depth: int = field(
        metadata={'help':'Decoder layer depths'}
    )
    num_heads: int = field(
        metadata={'help': 'transformer-based cross attention number of heads'}
    )
    mamba_version: str = field(
        metadata={'help':'Bi-directional Mamba version = v1,v2(mamba vision),v3(our)'}
    )
    learnable_params: bool = field(
        metadata={'help':'Differential Transformer Method in cross-attention'}
    )
    flash_attn: bool = field(
        metadata={'help':'In transformer-based cross attention using flash attention'}
    )
    mode: str = field(
        metadata={'help':'For Decoder positional encoding choose mode = Train, Inference'}
    )
    Linforemr: bool = field(
         metadata='choose using Linformer or not'
    )
    

def load_tokenizer(language=None,language_token=None):

    woptions = whisper.DecodingOptions(language,without_timestamps=True)

    tokenizer = whisper.tokenizer.get_tokenizer(False,language_token,task=woptions)
    return tokenizer

normalizer = BasicTextNormalizer()
tokenizer = load_tokenizer()




model = SpeechToTextModel(
    embed_dim=512,
    k_dim=256,
    depth=3,
    num_heads=8,
    mamba_version='v3',
    learnable_params=True,
    flash_attn=True,
    teacher_mode='base',
    mode='Inference'
)


metrics_wer = evaluate.load("wer")
metrics_cer = evaluate.load("cer")

# freeze whisper encoder
def set_trainable_parameters(module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad
        module._requires_grad = requires_grad


def main():
    metrics_wer = evaluate.load("wer")
    metrics_cer = evaluate.load("cer")
    model_args = ModelArguments()