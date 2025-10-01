import os.path

import torch
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.nanosvg import NanoSVGDataset
from typing import Optional
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer




def main(config:Optional[TrainingArguments]):

    # -------------- dataset -------------- #
    trainFile = os.path.join(os.path.dirname(__file__),"train.nano.parquet")
    testFile = os.path.join(os.path.dirname(__file__), "test.nano.parquet")
    trainDataset = NanoSVGDataset(trainFile)
    testDataset = NanoSVGDataset(testFile)

    # -------------- log+swanlib -------------- #

    # -------------- model & lora setup -------------- #

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
    )
    modelName = "starvector/starvector-1b-im2svg"

    modelSV = StarVectorForCausalLM.from_pretrained(modelName, torch_dtype=torch.float16, trust_remote_code=True)
    # 将 LoRA 应用于模型
    peftModel = get_peft_model(modelSV, config)
    peftModel.print_trainable_parameters()
    args = TrainingArguments(
        output_dir="output/i2v_1b_lora_sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=10,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        gradient_checkpointing=False,
        logging_dir="output/logs/i2v_1b_lora_sft",
    )

    trainer = Trainer(
        model=peftModel,
        args=args,
        train_dataset=trainDataset,  # 需要提供数据
        eval_dataset=testDataset,
    )
    # -------------- train -------------- #
    trainer.train()
    # -------------- model & lora setup -------------- #
    return