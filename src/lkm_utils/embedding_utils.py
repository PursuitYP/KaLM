import os
import torch
import logging
import numpy as np
from tqdm import tqdm


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    if attention_mask.device != token_embeddings.device:
        attention_mask = attention_mask.to(token_embeddings.device)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def eos_pooling(token_embeddings, attention_mask):
    if attention_mask.device != token_embeddings.device:
        attention_mask = attention_mask.to(token_embeddings.device)
    # get the last <eos> token
    last_ones_idx = torch.tensor([torch.where(mask == 1)[-1][-1] for mask in attention_mask])
    bs = token_embeddings.size()[0]
    token_embeddings = token_embeddings[torch.arange(bs), last_ones_idx, :]
    
    return token_embeddings.float()


def get_embedding_no_grad(model, tokenizer, sentences):
    """
    Get sentence embeddings from language model with mean pooling.
    Args:
        model: language model
        tokenizer: tokenizer
        sentences: list of sentences

    Returns: sentence embeddings
    """
    # remove <bos> and <eos> token
    if hasattr(tokenizer, "add_bos_token") and hasattr(tokenizer, "add_eos_token"):
        setattr(tokenizer, "add_bos_token", False) # for LLaMA tokenizer
        setattr(tokenizer, "add_eos_token", False)
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, return_tensors='pt')
    # Compute token embeddings, note that the model is prepared with a embed_head
    with torch.no_grad():
        _, _, sentence_embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True)

    return sentence_embeddings


def embed(model, tokenizer, output_dir, embed_batch_size, entityText_all, relationText_all):
    """
    Embed all entities and relations in the knowledge graph.

    Args:
        model: the model with embed_head to be evaluated
        tokenizer: tokenizer
        output_dir: the path to the data
        embed_batch_size: the batch size for embedding
        entityText_all: all entity text
        relationText_all: all relation text
    
    Returns: entity embeddings and relation embeddings
    """
    model.eval()

    # get all entity embeddings
    entity_embeddings = []
    for i in tqdm(range(0, len(entityText_all), embed_batch_size), desc="Embedding entities"):
        entity_embeddings.append(get_embedding_no_grad(model, tokenizer, entityText_all[i: min(i+embed_batch_size, len(entityText_all))]))
    entity_embeddings = torch.cat(entity_embeddings, dim=0)
    logging.info(f"entity embeddings shape: {entity_embeddings.shape}")
    np.save(os.path.join(output_dir, 'entity_embedding'), entity_embeddings.cpu().numpy())

    # get all relation embeddings
    relation_embeddings = []
    for i in tqdm(range(0, len(relationText_all), embed_batch_size), desc="Embedding relations"):
        relation_embeddings.append(get_embedding_no_grad(model, tokenizer, relationText_all[i: min(i+embed_batch_size, len(relationText_all))]))
    relation_embeddings = torch.cat(relation_embeddings, dim=0)
    logging.info(f"relation embeddings shape: {relation_embeddings.shape}")
    np.save(os.path.join(output_dir, 'relation_embedding'), relation_embeddings.cpu().numpy())

    return entity_embeddings, relation_embeddings


def get_embedding_no_grad_prob(model, tokenizer, sentences, pooling_type, sentence_type):
    """
    Get sentence embeddings from language model with mean pooling.
    Args:
        model: language model
        tokenizer: tokenizer
        sentences: list of sentences

    Returns: sentence embeddings
    """
    # remove <bos> and <eos> token
    if hasattr(tokenizer, "add_bos_token") and hasattr(tokenizer, "add_eos_token"):
        setattr(tokenizer, "add_bos_token", False) # for LLaMA tokenizer
        setattr(tokenizer, "add_eos_token", False)
    # Tokenize sentences
    if pooling_type == "mean":
        encoded_input = tokenizer(sentences, padding=True, return_tensors='pt')
    elif pooling_type == "eos":
        if sentence_type == "entity":   # add delimiter "[" + sentence + "]
            sentences = ["[ " + sentence + " ]" for sentence in sentences]
        elif sentence_type == "relation":   # add delimiter "{" + sentence + "}"
            sentences = ["{ " + sentence + " }" for sentence in sentences]
        encoded_input = tokenizer(sentences, padding=True, return_tensors='pt')

    # Compute token embeddings, note that the model is prepared with a embed_head
    with torch.no_grad():
        sentence_embeddings = model(**encoded_input, output_hidden_states=True, return_dict=True).hidden_states[-1]
    
    if pooling_type == "mean":
        sentence_embeddings = mean_pooling(sentence_embeddings, encoded_input["attention_mask"])
    elif pooling_type == "eos":
        sentence_embeddings = eos_pooling(sentence_embeddings, encoded_input["attention_mask"])

    return sentence_embeddings


def embed_prob(model, tokenizer, output_dir, embed_batch_size, entityText_all, relationText_all, pooling_type):
    """
    Embed all entities and relations in the knowledge graph.

    Args:
        model: the model with embed_head to be evaluated
        tokenizer: tokenizer
        output_dir: the path to the data
        embed_batch_size: the batch size for embedding
        entityText_all: all entity text
        relationText_all: all relation text
    
    Returns: entity embeddings and relation embeddings
    """
    model.eval()

    # get all relation embeddings
    relation_embeddings = []
    sentence_type = "relation"
    for i in tqdm(range(0, len(relationText_all), embed_batch_size), desc="Embedding relations"):
        relation_embeddings.append(get_embedding_no_grad_prob(model, tokenizer, relationText_all[i: min(i+embed_batch_size, len(relationText_all))], pooling_type, sentence_type))
    relation_embeddings = torch.cat(relation_embeddings, dim=0)
    logging.info(f"relation embeddings shape: {relation_embeddings.shape}")
    logging.info(f"relation embeddings type: {type(relation_embeddings)}")
    logging.info(f"relation embeddings dtype: {relation_embeddings.dtype}")
    np.save(os.path.join(output_dir, 'relation_embedding'), relation_embeddings.cpu().numpy())

    # get all entity embeddings
    entity_embeddings = []      # len(entityText_all)
    sentence_type = "entity"
    for i in tqdm(range(0, len(entityText_all), embed_batch_size), desc="Embedding entities"):
        entity_embeddings.append(get_embedding_no_grad_prob(model, tokenizer, entityText_all[i: min(i+embed_batch_size, len(entityText_all))], pooling_type, sentence_type))
    entity_embeddings = torch.cat(entity_embeddings, dim=0)
    logging.info(f"entity embeddings shape: {entity_embeddings.shape}")
    np.save(os.path.join(output_dir, 'entity_embedding'), entity_embeddings.cpu().numpy())

    return entity_embeddings, relation_embeddings
