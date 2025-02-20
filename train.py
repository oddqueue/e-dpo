import argparse
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import set_seed

from config import EpsilonDPOConfig
from trainer import EpsilonDPOTrainer


def main(config):
    set_seed(config.training_args.seed)

    model = AutoModelForCausalLM.from_pretrained(**config.model, torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(**config.model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    columns = ['chosen', 'rejected']
    dataset = load_dataset(config.dataset.name)
    train_dataset = dataset[config.dataset.train_split].select_columns(columns)
    if config.dataset.eval_split:
        eval_dataset = dataset[config.dataset.eval_split].select_columns(columns)
    else:
        eval_dataset = None

    training_args = EpsilonDPOConfig(**config.training_args)
    trainer = EpsilonDPOTrainer(model=model,
                                ref_model=ref_model,
                                args=training_args,
                                processing_class=tokenizer,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset)
    trainer.train()
    trainer.save_model(config.training_args.output_dir)
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='ε-DPO configuration')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='path to configuration',
    )
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    main(config)