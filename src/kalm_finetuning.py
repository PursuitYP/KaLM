import random
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
from datasets import Dataset
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from transformers.utils import PaddingStrategy

from llmtuner.extras.logging import get_logger
from llmtuner.extras.misc import count_parameters
from llmtuner.extras.callbacks import LogCallback, SavePeftModelCallback
from llmtuner.extras.ploting import plot_loss
from lkm_utils.dataset_utils import get_triples, get_text_info_inverse


# set logger
logger = get_logger(__name__)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the LKM training script.
    """
    # model parameters
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    output_dir: Optional[str] = field(default="llama_models/lkm", metadata={"help": "the output directory"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "whether to resume training from checkpoint"})

    # lora parameters
    lora_alpha: Optional[float] = field(default=16.0, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "the lora rank parameter"})
    
    # dataset parameters
    dataset_path: Optional[str] = field(default="data/wn18rr", metadata={"help": "the dataset path"})
    ent_cutoff_len: Optional[int] = field(default=50, metadata={"help": "the entity cutoff length"})
    sft_cutoff_len: Optional[int] = field(default=256, metadata={"help": "the sft cutoff length"})
    
    # kge parameters
    add_margin: Optional[float] = field(default=0.02, metadata={"help": "the add margin parameter for kge loss"})
    embedding_dim: Optional[int] = field(default=1000, metadata={"help": "the kg embedding dimension"})
    init_t: Optional[float] = field(default=0.05, metadata={"help": "the initial temperature for the lkm loss"})
    
    # training parameters
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    num_gpus: Optional[int] = field(default=4, metadata={"help": "the number of gpus for training"})
    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "the per device train batch size"})
    sft_batch_size: Optional[int] = field(default=4, metadata={"help": "the sft batch size"})
    negative_sample_size: Optional[int] = field(default=8, metadata={"help": "the negative sample size"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "whether to use gradient checkpointing"})
    bf16: Optional[bool] = field(default=True, metadata={"help": "whether to use bfloat16 mixed precision training"})
    save_steps: Optional[int] = field(default=1000, metadata={"help": "the steps to save the model"})
    log_steps: Optional[int] = field(default=10, metadata={"help": "the steps to log the training information"})
    negative_adversarial_sampling: Optional[bool] = field(default=False, metadata={"help": "whether to use negative adversarial sampling"})
    adversarial_temperature: Optional[float] = field(default=1.0, metadata={"help": "the adversarial temperature"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    weight_decay: Optional[float] = field(default=0.001, metadata={"help": "the weight decay"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer to use"})

    use_lm_loss: Optional[bool] = field(default=True, metadata={"help": "whether to use lm loss"})
    use_kge_loss: Optional[bool] = field(default=True, metadata={"help": "whether to use kge loss"})
    loss_alpha: Optional[float] = field(default=1.0, metadata={"help": "weight for the joint loss"})

    # test parameters
    test_batch_size: Optional[int] = field(default=8, metadata={"help": "the test batch size"})
    embedding_batch_size: Optional[int] = field(default=100, metadata={"help": "the embedding batch size"})


""" define the preprocess function and tokenize the dataset """
# define the preprocess function
def preprocess_function(examples):
    model_inputs = {
        "head_relation_input_ids": [],
        "head_relation_attention_mask": [],
        "tail_input_ids": [],
        "tail_attention_mask": [],
        "head_input_ids": [],
        "head_attention_mask": [],
        "sft_input_ids": [],
        "sft_attention_mask": [],
        "sft_labels": [],
        "head_id": [],
        "relation_id": [],
        "tail_id": [],
    }

    for positive_sample in examples["positive_sample"]:
        head, relation, tail = positive_sample
        
        ## disable the bos_token and eos_token for embedding training
        if hasattr(tokenizer, "add_bos_token") and hasattr(tokenizer, "add_eos_token"):
            setattr(tokenizer, "add_bos_token", False)
            setattr(tokenizer, "add_eos_token", False)
        head_text = entityText_all[head]
        relation_text = relationText_all[relation]
        tail_text = entityText_all[tail]

        head_text_original = entityText_all[head]
        tail_text_original = entityText_all[tail]

        ## cutoff length is set to script_args.ent_cutoff_len
        ent_cutoff_len = script_args.ent_cutoff_len
        head_tokenized = tokenizer(head_text, truncation=True, max_length=ent_cutoff_len)
        tail_tokenized = tokenizer(tail_text, truncation=True, max_length=ent_cutoff_len)
        head_input_ids = head_tokenized["input_ids"]
        tail_input_ids = tail_tokenized["input_ids"]
        head_text_cutoff = tokenizer.decode(head_input_ids)
        tail_text_cutoff = tokenizer.decode(tail_input_ids)
        ## add the hr_delimiter: [ head, relation ]
        head_relation_text = "[ " + head_text_cutoff + ", " + relation_text + " ]"
        ## add the tail_delimiter: { tail }
        head_text = "{ " + head_text_cutoff + " }"      # for self-negatives
        tail_text = "{ " + tail_text_cutoff + " }"      # for in-batch negatives
        ## tokenize the head_text, tail_text and head_relation_text
        head_tokenized = tokenizer(head_text, truncation=True, max_length=ent_cutoff_len+10)
        tail_tokenized = tokenizer(tail_text, truncation=True, max_length=ent_cutoff_len+10)
        head_relation_tokenized = tokenizer(head_relation_text, truncation=True, max_length=ent_cutoff_len+20)

        ## prepare for the sft prompt
        sft_cutoff_len = script_args.sft_cutoff_len
        prompt_point = {
            "instruction": "Given the head entity and relation, write a tail entity that completes the triple",
            "input": head_text_original + ", " + relation_text,
            "output": tail_text_original,
        }
        ## follow the alpaca instruct tuning prompt template
        user_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt_point["instruction"]}\n\n### Input:\n{prompt_point["input"]}\n\n### Response:\n"""
        ## enable the bos_token and eos_token for sft training
        if hasattr(tokenizer, "add_bos_token") and hasattr(tokenizer, "add_eos_token"):
            setattr(tokenizer, "add_bos_token", True)
            setattr(tokenizer, "add_eos_token", True)
        user_prompt_len = len(tokenizer(user_prompt, truncation=True, max_length=sft_cutoff_len)["input_ids"]) - 1 # don't take account the eos token
        full_tokens = tokenizer(user_prompt + prompt_point["output"], truncation=True, max_length=sft_cutoff_len)["input_ids"]

        model_inputs["sft_input_ids"].append(full_tokens)
        model_inputs["sft_attention_mask"].append([1] * len(full_tokens))
        model_inputs["sft_labels"].append([-100] * user_prompt_len + full_tokens[user_prompt_len:])
        model_inputs["head_relation_input_ids"].append(head_relation_tokenized["input_ids"])
        model_inputs["head_relation_attention_mask"].append(head_relation_tokenized["attention_mask"])
        model_inputs["tail_input_ids"].append(tail_tokenized["input_ids"])
        model_inputs["tail_attention_mask"].append(tail_tokenized["attention_mask"])
        model_inputs["head_input_ids"].append(head_tokenized["input_ids"])
        model_inputs["head_attention_mask"].append(head_tokenized["attention_mask"])
        model_inputs["head_id"].append(head)
        model_inputs["relation_id"].append(relation)
        model_inputs["tail_id"].append(tail)
    
    return model_inputs


""" define the LKM DataCollator that batches the data """
@dataclass
class LKMDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_head_relation = []
        features_tail = []
        features_head = []
        features_sft = []
        features_head_id = []
        features_relation_id = []
        features_tail_id = []

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the same length to return tensors.
        labels = [feature["sft_labels"] for feature in features]
        max_label_length = max(len(l) for l in labels)
        label_pad_token_id = -100
        for feature in features:
            remainder = [label_pad_token_id] * (max_label_length - len(feature["sft_labels"]))
            feature["sft_labels"] = feature["sft_labels"] + remainder

        for feature in features:
            features_head_relation.append(
                {
                    "input_ids": feature["head_relation_input_ids"],
                    "attention_mask": feature["head_relation_attention_mask"],
                }
            )
            features_tail.append(
                {
                    "input_ids": feature["tail_input_ids"],
                    "attention_mask": feature["tail_attention_mask"],
                }
            )
            features_head.append(
                {
                    "input_ids": feature["head_input_ids"],
                    "attention_mask": feature["head_attention_mask"],
                }
            )
            features_sft.append(
                {
                    "input_ids": feature["sft_input_ids"],
                    "attention_mask": feature["sft_attention_mask"],
                    "labels": feature["sft_labels"],
                }
            )
            features_head_id.append(feature["head_id"])
            features_relation_id.append(feature["relation_id"])
            features_tail_id.append(feature["tail_id"])

        # calculate head, relation and tail ids in this batch
        head_ids = torch.LongTensor(features_head_id)
        relation_ids = torch.LongTensor(features_relation_id)
        tail_ids = torch.LongTensor(features_tail_id)
        # initialize the triple_mask matrix
        bs = len(head_ids)
        triple_mask = (tail_ids.unsqueeze(1) != tail_ids.unsqueeze(0))
        triple_mask.fill_diagonal_(True)    # all True
        # mask out the in-batch positive samples
        for i in range(bs):     # for each row, head-relation
            head_id, relation_id = head_ids[i], relation_ids[i]
            neighbor_tail_ids = hr2t.get((head_id, relation_id), set())
            # exact match is enough, no further check needed
            if len(neighbor_tail_ids) <= 1:
                continue
            for j in range(bs):     # for each column, tail
                if i == j:
                    continue
                tail_id = tail_ids[j]
                if tail_id in neighbor_tail_ids:
                    # mask out positive samples
                    triple_mask[i][j] = False

        batch_head_relation = self.tokenizer.pad(
            features_head_relation,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_tail = self.tokenizer.pad(
            features_tail,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_head = self.tokenizer.pad(
            features_head,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_sft = self.tokenizer.pad(
            features_sft,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "head_relation_input_ids": batch_head_relation["input_ids"],
            "head_relation_attention_mask": batch_head_relation["attention_mask"],
            "tail_input_ids": batch_tail["input_ids"],
            "tail_attention_mask": batch_tail["attention_mask"],
            "head_input_ids": batch_head["input_ids"],
            "head_attention_mask": batch_head["attention_mask"],
            "sft_input_ids": batch_sft["input_ids"],
            "sft_attention_mask": batch_sft["attention_mask"],
            "sft_labels": batch_sft["labels"],
            "triple_mask": triple_mask,
        }

        return batch


""" define LKMTrainer and rewrite the compute_loss function """
class LKMTrainer(Trainer):
    # define how to compute the lkm loss
    '''
    inputs: a batch of inputs
        head_relation_input_ids: a batch of input_ids for head_relation texts
        head_relation_attention_mask: a batch of attention_masks for head_relation texts
        tail_input_ids: a batch of input_ids for tail entities
        tail_attention_mask: a batch of attention_masks for tail entities
        head_input_ids: a batch of input_ids for head entities
        head_attention_mask: a batch of attention_masks for head entities
        sft_input_ids: a batch of input_ids for sft texts
        sft_attention_mask: a batch of attention_masks for sft texts
        sft_labels: a batch of labels for sft texts
        triple_mask: a batch of triple_mask for this batch
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # re-parameterize log(1 / \tau) as a learnable parameter: temperature in InfoNCE loss
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / script_args.init_t).log(), requires_grad=True)

    def compute_loss(self, model, inputs):
        # for debug usage
        # logger.info(f"****** modes in this batch ### {mode} ###")
        lm_loss = 0
        kge_loss = 0
        """ 1. compute the sft lm loss """
        if script_args.use_lm_loss:
            # select a subset of the batch for sft lm loss (avoid OOM)
            lm_index = random.sample(range(0, script_args.per_device_train_batch_size), script_args.sft_batch_size)
            sft_input_ids_subset = inputs["sft_input_ids"][lm_index]
            sft_attention_mask_subset = inputs["sft_attention_mask"][lm_index]
            sft_labels_subset = inputs["sft_labels"][lm_index]
            lm_loss= model(input_ids=sft_input_ids_subset, attention_mask=sft_attention_mask_subset, labels=sft_labels_subset, return_dict=True).loss

        """ 2. compute the kge loss """
        if script_args.use_kge_loss:
            # get head_relation, tail and head embeddings (last hidden state, unpooled): ## [bs, seq_len, 4096]
            head_relation_embedding = model(input_ids=inputs["head_relation_input_ids"], attention_mask=inputs["head_relation_attention_mask"], output_hidden_states=True, return_dict=True).hidden_states[-1]
            tail_embedding = model(input_ids=inputs["tail_input_ids"], attention_mask=inputs["tail_attention_mask"], output_hidden_states=True, return_dict=True).hidden_states[-1]
            head_embedding = model(input_ids=inputs["head_input_ids"], attention_mask=inputs["head_attention_mask"], output_hidden_states=True, return_dict=True).hidden_states[-1]

            # get head_relation, tail and head embeddings from the [EOS] token (not the last token): ## [bs, 4096]
            head_relation_attention_mask = inputs["head_relation_attention_mask"]
            last_ones_idx = torch.tensor([torch.where(mask == 1)[-1][-1] for mask in head_relation_attention_mask])
            head_relation_embedding = head_relation_embedding[torch.arange(script_args.per_device_train_batch_size), last_ones_idx, :]
            head_attention_mask = inputs["head_attention_mask"]
            last_ones_idx = torch.tensor([torch.where(mask == 1)[-1][-1] for mask in head_attention_mask])
            head_embedding = head_embedding[torch.arange(script_args.per_device_train_batch_size), last_ones_idx, :]
            tail_attention_mask = inputs["tail_attention_mask"]
            last_ones_idx = torch.tensor([torch.where(mask == 1)[-1][-1] for mask in tail_attention_mask])
            tail_embedding = tail_embedding[torch.arange(script_args.per_device_train_batch_size), last_ones_idx, :]

            # make the embeddings L2 normalized
            head_relation_embedding = F.normalize(head_relation_embedding, p=2, dim=1)
            tail_embedding = F.normalize(tail_embedding, p=2, dim=1)
            head_embedding = F.normalize(head_embedding, p=2, dim=1)

            # labales for the InfoNCE loss: [0, 1, 2, ..., batch_size-1]
            labels = torch.arange(script_args.per_device_train_batch_size).to(head_relation_embedding.device)

            # calculate logits for positive samples and in-batch negative samples
            logits = head_relation_embedding.mm(tail_embedding.t())
            logits -= torch.zeros(logits.size()).fill_diagonal_(script_args.add_margin).to(logits.device)
            logits *= self.log_inv_t.exp()      ## [bs, bs]

            # get the triple_mask matrix, mask out the in-batch positive samples
            triple_mask = inputs["triple_mask"].to(logits.device)
            logits.masked_fill_(~triple_mask, -1e4)
            
            # calculate the InfoNCE loss: row-wise (hr->t) and column-wise(t->hr)
            criterion = torch.nn.CrossEntropyLoss().cuda()
            kge_loss = (criterion(logits, labels) + criterion(logits.t(), labels)) / 2

        """ 3. compute the final loss """
        loss = script_args.use_kge_loss * kge_loss + script_args.loss_alpha * script_args.use_lm_loss * lm_loss

        return loss


if __name__ == "__main__":
    # parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]


    """ 1. load model and tokenizer """
    # load model
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
        use_cache=False,
    )

    # load peft config
    peft_config = LoraConfig(
        r=script_args.lora_rank,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, peft_config)


    # print the model parameters
    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    # print the model parameters (detailed)
    for name, param in model.named_parameters():
        logger.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token


    """ 2. load and preprocess dataset """
    """ 2.1 load dataset from local files """
    # read the dataset (triple information)
    train_triples, valid_triples, test_triples, all_true_triples = get_triples(script_args.dataset_path)
    logger.info(f"train triples: {len(train_triples)}, valid triples: {len(valid_triples)}, test triples: {len(test_triples)}")
    logger.info(f"all true triples: {len(all_true_triples)}")
    # read the dataset (text information)
    # entityText_all, relationText_all = get_text_info(script_args.dataset_path)
    entityText_all, relationText_all, relationText_inverse_all = get_text_info_inverse(script_args.dataset_path)
    logger.info(f"entity count: {len(entityText_all)}, relation count: {len(relationText_all)}")
    nentity = len(entityText_all)
    nrelation = len(relationText_all)

    # add inverse relations to train triples: (head, relation, tail) => (tail, relation + nrelation, head)
    train_triples += [(tail, relation + nrelation, head) for head, relation, tail in train_triples]
    logger.info(f"## train triples (inverse relation augmented): {len(train_triples)}")
    # add inverse relations to relationText_all: relationText_all => relationText_all + relationText_inverse_all
    relationText_all.extend(relationText_inverse_all)
    logger.info(f"## relation count (inverse relation augmented): {len(relationText_all)}")
    logger.info(f"## relation texts (inverse relation augmented): {relationText_all}")

    # calculate the head-relation to tails dictionary (hr2t)
    hr2t = {}
    ## iterate over the train triples
    for head, relation, tail in train_triples:
        if (head, relation) not in hr2t:
            hr2t[(head, relation)] = []
        hr2t[(head, relation)].append(tail)
    ## de-duplicate the tail entities with set()
    for head, relation in hr2t:
        hr2t[(head, relation)] = np.array(list(set(hr2t[(head, relation)])))

    """ 2.2 convert the train dataset to a huggingface dataset (<class 'datasets.arrow_dataset.Dataset'>) """
    data_dict = {
        "positive_sample": [],
    }

    for positive_sample in train_triples:
        data_dict["positive_sample"].append(positive_sample)

    # load datasets.arrow_dataset.Dataset from dict
    train_dataset_hf = Dataset.from_dict(data_dict)
    logger.info(f"train_dataset_hf length: {len(train_dataset_hf['positive_sample'])}")

    """ 2.3 preprocess the train dataset with the preprocess function """
    original_columns = train_dataset_hf.column_names
    train_dataset_hf = train_dataset_hf.map(
        preprocess_function, 
        batched=True, 
        num_proc=24, 
        remove_columns=original_columns,
    )


    """ 3. define the training arguments and start the Trainer """
    # define the training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        remove_unused_columns=False,
        bf16=script_args.bf16,
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        lr_scheduler_type=script_args.lr_scheduler_type,
        optim=script_args.optim,
        logging_steps=script_args.log_steps,
        save_steps=script_args.save_steps,
        num_train_epochs=script_args.num_train_epochs,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


    # start the Trainer
    trainer = LKMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset_hf,
        data_collator=LKMDataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[LogCallback()] + [SavePeftModelCallback()],
    )

    trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

    logger.info("Saving last checkpoint of the model")
    trainer.save_state()
    trainer.save_model()
    if trainer.is_world_process_zero():
        plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])
