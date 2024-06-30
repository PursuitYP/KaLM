import os


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.

    Args:
        file_path: path to the dataset
        entity2id: dict of entity mapping
        relation2id: dict of relation mapping
    
    Returns: list of triples
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def get_triples(data_path):
    """
    Read train, valid and test triples from dataset.

    Args:
        data_path: path to the dataset
    
    Returns: train, valid and test triples, and all true triples
    """
    # Read entity mapping
    with open(os.path.join(data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    # Read relation mapping
    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    # Read train, valid and test triples
    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)

    # All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    return train_triples, valid_triples, test_triples, all_true_triples


def get_text_info(data_path):
    """
    Read entity text information and relation text information from dataset.

    Args:
        data_path: path to the dataset

    Returns: entity text information and relation text information
    """
    # Read additional entity text information
    with open(os.path.join(data_path, 'entityId2text.dict')) as fin:
        entityText_all = []
        for line in fin:
            eid, entityText = line.strip().split('\t')
            entityText_all.append(entityText)

    # Read additional relation text information
    with open(os.path.join(data_path, 'relationId2text.dict')) as fin:
        relationText_all = []
        for line in fin:
            rid, relationText = line.strip().split('\t')
            relationText_all.append(relationText)

    return entityText_all, relationText_all


def get_text_info_inverse(data_path):
    """
    Read entity text information and relation text information from dataset.

    Args:
        data_path: path to the dataset

    Returns: entity text information and relation text information
    """
    # Read additional entity text information
    with open(os.path.join(data_path, 'entityId2text.dict')) as fin:
        entityText_all = []
        for line in fin:
            eid, entityText = line.strip().split('\t')
            entityText_all.append(entityText)

    # Read additional relation text information
    with open(os.path.join(data_path, 'relationId2text.dict')) as fin:
        relationText_all = []
        for line in fin:
            rid, relationText = line.strip().split('\t')
            relationText_all.append(relationText)
    
    # Read additional inverse relation text information
    with open(os.path.join(data_path, 'relationId2text_inverse.dict')) as fin:
        relationText_inverse_all = []
        for line in fin:
            rid, relationText_inverse = line.strip().split('\t')
            relationText_inverse_all.append(relationText_inverse)

    return entityText_all, relationText_all, relationText_inverse_all


def get_text_info_long(data_path):
    """
    Read entity text information and relation text information from dataset.

    Args:
        data_path: path to the dataset

    Returns: entity text information and relation text information
    """
    # Read additional entity text information
    with open(os.path.join(data_path, 'entityId2textlong.dict')) as fin:
        entityText_all = []
        for line in fin:
            eid, entityText = line.strip().split('\t')
            entityText_all.append(entityText)

    # Read additional relation text information
    with open(os.path.join(data_path, 'relationId2text.dict')) as fin:
        relationText_all = []
        for line in fin:
            rid, relationText = line.strip().split('\t')
            relationText_all.append(relationText)

    return entityText_all, relationText_all
