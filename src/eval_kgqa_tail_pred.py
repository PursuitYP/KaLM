import os
import logging
import fire
import torch
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel
from tqdm import tqdm
from lkm_utils.dataset_utils import get_triples, get_text_info


def set_logger(output_dir: str, log_name: str):
    '''
    Write logs to checkpoint and console
    '''
    log_file = os.path.join(output_dir, log_name)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main(
        base_model: str = "meta-llama/Llama-2-7b-hf", 
        lora_ckpt_dir: str = "llama_models/llama-2-7b", 
        dataset_dir: str = "data/wn18rr", 
        with_desc: bool = True     # whether to use entity descriptions
        ):
    # set logger
    if with_desc:
        set_logger(lora_ckpt_dir, 'kgqa_tail_pred_wdecs.log')
    else:
        set_logger(lora_ckpt_dir, 'kgqa_tail_pred_wodecs.log')

    # load model and tokenizer
    if lora_ckpt_dir != "llama_models/llama-2-7b":
        logging.info(f"# model checkpoint directory: {lora_ckpt_dir}")
    else:
        logging.info(f"# evaluate on base model only: {base_model}")

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )

    if lora_ckpt_dir != "llama_models/llama-2-7b":
        model = PeftModel.from_pretrained(model, lora_ckpt_dir)
        model = model.merge_and_unload()
        logging.info("Successfully loaded model lora weight.")
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()


    # load dataset
    train_triples, valid_triples, test_triples, all_true_triples = get_triples(dataset_dir)
    logging.info(f"train triples: {len(train_triples)}, valid triples: {len(valid_triples)}, test triples: {len(test_triples)}")
    logging.info(f"all true triples: {len(all_true_triples)}")
    # read the dataset (text information)
    entityText_all, relationText_all = get_text_info(dataset_dir)
    logging.info(f"entity count: {len(entityText_all)}, relation count: {len(relationText_all)}")


    """ construct and save the prompt-response pairs (ground truth) """
    # concatenate all entity texts
    relation_concat = ""
    for rel in relationText_all:
        relation_concat += rel + " | "
    relation_concat = relation_concat[:-3]
    prompt, response = [], []
    for head_id, relation_id, tail_id in tqdm(test_triples, desc="construct prompt-response pairs"):
        # construct the prompt
        head_text = entityText_all[head_id]
        relation_text = relationText_all[relation_id]
        tail_text = entityText_all[tail_id]

        # only keep text before the first ", " (if any)
        head_text = head_text.split(", ")[0]
        tail_text = tail_text.split(", ")[0]
    
        prompt_point = f"What is the relation between [{head_text}] and [{tail_text}]?"
        response_point = relation_text
        prompt.append(prompt_point)
        response.append(response_point)

    output = pd.DataFrame({'prompt': prompt, 'response': response})
    output.to_csv(os.path.join(lora_ckpt_dir, "prompt_response_gt.csv"), header=True, index=False)
    logging.info(f"Successfully saved prompt-response pairs to {os.path.join(lora_ckpt_dir, 'prompt_response_gt.csv')}")


    """ construct ICL examples """
    # construct the prompt_point_example and pred_response_example from the 1-st triple in train_triples
    head_id_example, relation_id_example, tail_id_example = train_triples[0]
    head_text_example = entityText_all[head_id_example]
    relation_text_example = relationText_all[relation_id_example]
    tail_text_example = entityText_all[tail_id_example]
    head_text_example = head_text_example.split("; ")[0]
    tail_text_example = tail_text_example.split("; ")[0]
    if head_text_example.find("etc.") != -1:
        head_text_example = head_text_example[: -6]
    if tail_text_example.find("etc.") != -1:
        tail_text_example = tail_text_example[: -6]
    if not with_desc:
        head_text_example = head_text_example.split(", ")[0]
        tail_text_example = tail_text_example.split(", ")[0]
        prompt_point_example_zero = f"Given the head entity and relation, write a tail entity that completes the triple: {head_text_example} {relation_text_example}"
    else:
        prompt_point_example_zero = f"Given the head entity and relation, write a tail entity that completes the triple: {head_text_example}, {relation_text_example}"
    pred_response_example_zero = tail_text_example
    # construct the prompt_point_example and pred_response_example from the 18-th triple in train_triples
    head_id_example, relation_id_example, tail_id_example = train_triples[17]
    head_text_example = entityText_all[head_id_example]
    relation_text_example = relationText_all[relation_id_example]
    tail_text_example = entityText_all[tail_id_example]
    head_text_example = head_text_example.split("; ")[0]
    tail_text_example = tail_text_example.split("; ")[0]
    if head_text_example.find("etc.") != -1:
        head_text_example = head_text_example[: -6]
    if tail_text_example.find("etc.") != -1:
        tail_text_example = tail_text_example[: -6]
    if not with_desc:
        head_text_example = head_text_example.split(", ")[0]
        tail_text_example = tail_text_example.split(", ")[0]
        prompt_point_example_one = f"Given the head entity and relation, write a tail entity that completes the triple: {head_text_example} {relation_text_example}"
    else:
        prompt_point_example_one = f"Given the head entity and relation, write a tail entity that completes the triple: {head_text_example}, {relation_text_example}"
    pred_response_example_one = tail_text_example
    # construct the prompt_point_example and pred_response_example from the 131-th triple in train_triples
    head_id_example, relation_id_example, tail_id_example = train_triples[130]
    head_text_example = entityText_all[head_id_example]
    relation_text_example = relationText_all[relation_id_example]
    tail_text_example = entityText_all[tail_id_example]
    head_text_example = head_text_example.split("; ")[0]
    tail_text_example = tail_text_example.split("; ")[0]
    if head_text_example.find("etc.") != -1:
        head_text_example = head_text_example[: -6]
    if tail_text_example.find("etc.") != -1:
        tail_text_example = tail_text_example[: -6]
    if not with_desc:
        head_text_example = head_text_example.split(", ")[0]
        tail_text_example = tail_text_example.split(", ")[0]
        prompt_point_example_two = f"Given the head entity and relation, write a tail entity that completes the triple: {head_text_example} {relation_text_example}"
    else:
        prompt_point_example_two = f"Given the head entity and relation, write a tail entity that completes the triple: {head_text_example}, {relation_text_example}"
    pred_response_example_two = tail_text_example


    """ construct the final prompt and do inference """
    prompt, pred_response = [], []
    cnt = 1
    for head_id, relation_id, tail_id in tqdm(test_triples, desc="inference on test triples"):
        # construct the prompt
        head_text = entityText_all[head_id]
        relation_text = relationText_all[relation_id]
        tail_text = entityText_all[tail_id]
        head_text = head_text.split("; ")[0]
        tail_text = tail_text.split("; ")[0]
        if head_text.find("etc.") != -1:
            head_text = head_text[: -6]
        if tail_text.find("etc.") != -1:
            tail_text = tail_text[: -6]
        if not with_desc:
            head_text = head_text.split(", ")[0]
            tail_text = tail_text.split(", ")[0]
            prompt_point = f"Given the head entity and relation, write a tail entity that completes the triple: {head_text} {relation_text}"
        else:
            prompt_point = f"Given the head entity and relation, write a tail entity that completes the triple: {head_text}, {relation_text}"

        # construct the prompt with template and do inference
        user_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
------
Here is an example:
### Instruction:
{prompt_point_example_zero}
### Response:
{pred_response_example_zero}
------
Here is another example:
### Instruction:
{prompt_point_example_one}
### Response:
{pred_response_example_one}
------
Here is the third example:
### Instruction:
{prompt_point_example_two}
### Response:
{pred_response_example_two}
------
Here is the task:
### Instruction:
{prompt_point}
### Response:"""

        # print(user_prompt)
        # break

        inputs = tokenizer(user_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        with torch.no_grad():
            generate_ids = model.generate(input_ids=input_ids, max_new_tokens=10)
        response_point = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # append to prompt and pred_response
        # prompt.append(prompt_point)
        prompt.append(user_prompt)
        pred_response.append(response_point)

        # logging.info(f"pred response: {response_point}")
        # if cnt > 2:
        #     break
        # cnt += 1
    
    output = pd.DataFrame({'prompt': prompt, 'pred_response': pred_response})
    if with_desc:
        output.to_csv(os.path.join(lora_ckpt_dir, "results_tail_pred_wdecs.csv"), header=True, index=False)
        logging.info(f"Successfully saved results to {os.path.join(lora_ckpt_dir, 'results_tail_pred_wdecs.csv')}")
    else:
        output.to_csv(os.path.join(lora_ckpt_dir, "results_tail_pred_wodecs.csv"), header=True, index=False)
        logging.info(f"Successfully saved results to {os.path.join(lora_ckpt_dir, 'results_tail_pred_wodecs.csv')}")
        

if __name__ == "__main__":
    fire.Fire(main)