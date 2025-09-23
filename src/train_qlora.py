import torch
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig 
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
from src.data_collator import CustomDataCollator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


    
def train_model(eval_ds, train_ds, processor, custom_data_collator):
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    save_dir = './qwen2-audio-7B-content'
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto"
    )
        
    qlora_args = SFTConfig(
        output_dir=save_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=1000,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=0.001,
        bf16=True,
        report_to="none",
        packing=False,
        gradient_checkpointing= True,
        #dataset_text_field="messages",
        push_to_hub = True
    )
    
    
    qlora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
        
    
    trainer_qlora = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=qlora_args,
        peft_config=qlora_config,
       data_collator=custom_data_collator
    )
    
    trainer_qlora.train()
    return model