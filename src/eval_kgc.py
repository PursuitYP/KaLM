import os
import logging
import fire
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

from lkm_utils.dataset_utils import get_triples, get_text_info_inverse
from lkm_utils.dataloader import TestDataset
from lkm_utils.embedding_utils import mean_pooling, eos_pooling


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


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(
        # model and lora parameters
        base_model: str = "meta-llama/Llama-2-7b-hf", 
        output_dir: str = "llama_models/llama-2-7b", 
        # dataset parameters
        data_path: str = "data/wn18rr", 
        cpu_num: int = 10,
        # training parameters
        max_steps: int = 1000,
        # test parameters
        test_batch_size: int = 8,
        embed_batch_size: int = 64,
        test_log_steps: int = 1000,
        # pooling type
        pooling_type: str = "mean",
        ent_cutoff_len: int = 50,
        ):
    """
    Method:
    1. load the model and tokenizer
    2. load lora config and get the peft model
    3. prepare the model with embed_head
    4. load the dataset with triples and texts 
    5. define the loss function and the train loop
    6. train the model with the train loop
    7. evaluate the model with the test loop
    8. save the model
    """
    # create output_dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set logger
    set_logger(output_dir, 'test.log')


    """ read and process the dataset """
    # read the dataset (triple information)
    train_triples, valid_triples, test_triples, all_true_triples = get_triples(data_path)
    logging.info(f"train triples: {len(train_triples)}, valid triples: {len(valid_triples)}, test triples: {len(test_triples)}")
    logging.info(f"all true triples: {len(all_true_triples)}")
    # read the dataset (text information)
    # entityText_all, relationText_all = get_text_info(data_path)
    entityText_all, relationText_all, relationText_inverse_all = get_text_info_inverse(data_path)
    logging.info(f"entity count: {len(entityText_all)}, relation count: {len(relationText_all)}")
    nentity = len(entityText_all)
    nrelation = len(relationText_all)

    # inverse the triples (entityText_all remains the same, relationText_all with the "inverse " prefix)
    train_triples_inverse = [(tail, relation, head) for head, relation, tail in train_triples]
    valid_triples_inverse = [(tail, relation, head) for head, relation, tail in valid_triples]
    test_triples_inverse = [(tail, relation, head) for head, relation, tail in test_triples]
    all_true_triples_inverse = [(tail, relation, head) for head, relation, tail in all_true_triples]
    # relationText_all_inverse = ["inverse " + relationText for relationText in relationText_all]
    relationText_all_inverse = relationText_inverse_all


    """ load the model """
    logging.info(f"# model checkpoint directory: {output_dir}")
    # load model from output_dir
    checkpoint_dir = output_dir
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if hasattr(tokenizer, "add_bos_token") and hasattr(tokenizer, "add_eos_token"):
        setattr(tokenizer, "add_bos_token", False) # for LLaMA tokenizer
        setattr(tokenizer, "add_eos_token", False)

    # set tokenizer padding token to eos_token, for padding=True setting
    tokenizer.pad_token = tokenizer.eos_token

    if checkpoint_dir != "llama_models/llama-2-7b":
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        model = model.merge_and_unload()
        logging.info("Successfully loaded model lora weight.")
    else:
        logging.info(f"# evaluate on base model only: {base_model}")
    
    model.eval()


    """ compute entity embeddings """
    if os.path.exists(os.path.join(output_dir, 'entity_embedding.npy')):
        logging.info("Load entity embeddings from cached file.")
        entity_embeddings = torch.from_numpy(np.load(os.path.join(output_dir, 'entity_embedding.npy')))
        entity_embeddings = entity_embeddings.cuda()
    else:
        logging.info("Get entity embeddings...")
        entity_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(entityText_all), embed_batch_size), desc="Embedding entities"):
                sentences = entityText_all[i: min(i+embed_batch_size, len(entityText_all))]
                # tokenize sentences
                if pooling_type == "mean":
                    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=ent_cutoff_len, return_tensors="pt")
                elif pooling_type == "eos":
                    # pre-tokenize the sentences
                    sentences_new = []
                    for i in range(len(sentences)):
                        encoded_one = tokenizer(sentences[i], truncation=True, max_length=ent_cutoff_len, return_tensors="pt")
                        sentences_new.append(tokenizer.decode(encoded_one["input_ids"][0]))
                    # add delimiter: "{ " + sentence + " }"
                    sentences = ["{ " + sentence + " }" for sentence in sentences_new]
                    # re-tokenize the sentences
                    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=ent_cutoff_len+10, return_tensors="pt")

                # compute token embeddings
                sentence_embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).hidden_states[-1]
                if pooling_type == "mean":
                    sentence_embeddings = mean_pooling(sentence_embeddings, encoded_input["attention_mask"])
                elif pooling_type == "eos":
                    sentence_embeddings = eos_pooling(sentence_embeddings, encoded_input["attention_mask"])
                entity_embeddings.append(sentence_embeddings)

        entity_embeddings = torch.cat(entity_embeddings, dim=0).cuda()
        np.save(os.path.join(output_dir, 'entity_embedding'), entity_embeddings.cpu().numpy())

    logging.info(f"entity embeddings shape: {entity_embeddings.shape}")
    entity_embeddings = F.normalize(entity_embeddings, p=2, dim=1)
    

    # load the test dataset with dataloader
    test_dataloader_tail = DataLoader(
        TestDataset(test_triples, all_true_triples, nentity, nrelation, 'tail-batch'),
        batch_size=test_batch_size,
        num_workers=max(1, cpu_num),
        collate_fn=TestDataset.collate_fn
    )
    test_dataloader_inverse = DataLoader(
        TestDataset(test_triples_inverse, all_true_triples_inverse, nentity, nrelation, 'tail-batch'),
        batch_size=test_batch_size,
        num_workers=max(1, cpu_num),
        collate_fn=TestDataset.collate_fn
    )

    # 1. evaluate the model on the test set
    logs = []
    step = 0
    with torch.no_grad():
        for positive_sample, negative_sample, filter_bias, mode in tqdm(test_dataloader_tail, desc="Evaluating on test set"):
            batch_size = positive_sample.size(0)
            head_texts = [entityText_all[head] for head in positive_sample[:, 0]]
            relation_texts = [relationText_all[relation] for relation in positive_sample[:, 1]]
            head_relation_texts = []
            # pre-tokenize the head entities
            for i in range(batch_size): 
                head_tokenized = tokenizer(head_texts[i], truncation=True, max_length=ent_cutoff_len, return_tensors="pt")
                head_relation_texts.append(tokenizer.decode(head_tokenized["input_ids"][0]) + ", " + relation_texts[i])
            # tokenize head_relation_texts
            if pooling_type == "mean":
                encoded_input = tokenizer(head_relation_texts, padding=True, truncation=True, max_length=ent_cutoff_len+20, return_tensors="pt")
            elif pooling_type == "eos":
                # add delimiter: "[ " + head + relation + " ]"
                head_relation_texts = ["[ " + head_relation_text + " ]" for head_relation_text in head_relation_texts]
                # re-tokenize the sentences
                encoded_input = tokenizer(head_relation_texts, padding=True, truncation=True, max_length=ent_cutoff_len+20, return_tensors="pt")
            # compute token embeddings
            head_relation_embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).hidden_states[-1]
            if pooling_type == "mean":
                head_relation_embeddings = mean_pooling(head_relation_embeddings, encoded_input["attention_mask"])
            elif pooling_type == "eos":
                head_relation_embeddings = eos_pooling(head_relation_embeddings, encoded_input["attention_mask"])
    
            # get the score, the higher the better
            head_relation_embeddings = head_relation_embeddings.cuda()
            head_relation_embeddings = F.normalize(head_relation_embeddings, p=2, dim=1)
            score = head_relation_embeddings.mm(entity_embeddings.t())
            filter_bias = filter_bias.cuda()
            score += filter_bias    # [bs, nentity]
            # explicitly sort all the entities to ensure that there is no test exposure bias
            argsort = torch.argsort(score, dim=1, descending=True)
            positive_arg = positive_sample[:, 2]
            for i in range(batch_size):
                # notice that argsort is not ranking
                ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                assert ranking.size(0) == 1
                # ranking + 1 is the true ranking used in evaluation metrics
                ranking = 1 + ranking.item()
                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0,
                })
            if step % test_log_steps == 0:
                logging.info('Evaluating the model on test set... (%d/%d)' % (step, len(test_dataloader_tail)))
            step += 1
    
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    log_metrics('Test result on test set', max_steps-1, metrics)


    # 2. evaluate the model on the test set (inverse)
    logs_inverse = []
    step = 0
    with torch.no_grad():
        for positive_sample, negative_sample, filter_bias, mode in tqdm(test_dataloader_inverse, desc="Evaluating on test set (inverse)"):
            batch_size = positive_sample.size(0)
            head_texts = [entityText_all[head] for head in positive_sample[:, 0]]
            relation_texts = [relationText_all_inverse[relation] for relation in positive_sample[:, 1]]
            head_relation_texts = []
            # pre-tokenize the head entities
            for i in range(batch_size):
                head_tokenized = tokenizer(head_texts[i], truncation=True, max_length=ent_cutoff_len, return_tensors="pt")
                head_relation_texts.append(tokenizer.decode(head_tokenized["input_ids"][0]) + ", " + relation_texts[i])
            # tokenize head_relation_texts
            if pooling_type == "mean":
                encoded_input = tokenizer(head_relation_texts, padding=True, truncation=True, max_length=ent_cutoff_len+20, return_tensors="pt")
            elif pooling_type == "eos":
                # add delimiter: "[ " + head + relation + " ]"
                head_relation_texts = ["[ " + head_relation_text + " ]" for head_relation_text in head_relation_texts]
                # re-tokenize the sentences
                encoded_input = tokenizer(head_relation_texts, padding=True, truncation=True, max_length=ent_cutoff_len+20, return_tensors="pt")
            # compute token embeddings
            head_relation_embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).hidden_states[-1]
            if pooling_type == "mean":
                head_relation_embeddings = mean_pooling(head_relation_embeddings, encoded_input["attention_mask"])
            elif pooling_type == "eos":
                head_relation_embeddings = eos_pooling(head_relation_embeddings, encoded_input["attention_mask"])
    
            # get the score, the higher the better
            head_relation_embeddings = head_relation_embeddings.cuda()
            head_relation_embeddings = F.normalize(head_relation_embeddings, p=2, dim=1)
            score = head_relation_embeddings.mm(entity_embeddings.t())
            filter_bias = filter_bias.cuda()
            score += filter_bias    # [bs, nentity]
            # explicitly sort all the entities to ensure that there is no test exposure bias
            argsort = torch.argsort(score, dim=1, descending=True)
            positive_arg = positive_sample[:, 2]
            for i in range(batch_size):
                # notice that argsort is not ranking
                ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                assert ranking.size(0) == 1
                # ranking + 1 is the true ranking used in evaluation metrics
                ranking = 1 + ranking.item()
                logs_inverse.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0,
                })
            if step % test_log_steps == 0:
                logging.info('Evaluating the model on test set (inverse)... (%d/%d)' % (step, len(test_dataloader_inverse)))
            step += 1
    
    metrics = {}
    for metric in logs_inverse[0].keys():
        metrics[metric] = sum([log_inverse[metric] for log_inverse in logs_inverse]) / len(logs_inverse)
    log_metrics('Test result on test set (inverse)', max_steps-1, metrics)


    # 3. all test results (merge the two logs list)
    logs_all = logs + logs_inverse
    metrics = {}
    for metric in logs_all[0].keys():
        metrics[metric] = sum([log_all[metric] for log_all in logs_all]) / len(logs_all)
    log_metrics('Test result on test set (all)', max_steps-1, metrics)


if __name__ == "__main__":
    fire.Fire(main)