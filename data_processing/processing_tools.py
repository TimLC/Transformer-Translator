import numpy as np
from torch.utils.data import DataLoader

def limit_size_text_dataset(dataset_first, dataset_second, max_lenght = 500):
    dataset = [[sentence_first, sentence_second] for sentence_first, sentence_second in zip(dataset_first, dataset_second) if
               len(sentence_first) <= max_lenght and len(sentence_second) <= max_lenght]
    dataset = list(zip(*dataset))
    dataset_first, dataset_second = list(dataset[0]), list(dataset[1])
    return dataset_first, dataset_second

def remove_duplicate(dataset_first, dataset_second):
    dataset = zip(dataset_first, dataset_second)
    dataset = list(dict.fromkeys(dataset))
    # dataset = list(set(dataset)) # Remove mélange
    dataset = list(zip(*dataset))
    dataset_first, dataset_second = list(dataset[0]), list(dataset[1])
    return dataset_first, dataset_second

def create_dataset_model(tokenizer_src, tokenizer_tgt, batch_size):
    source = np.zeros((len(tokenizer_src.sentences), tokenizer_src.max_length)).astype(np.int64)
    target = np.zeros((len(tokenizer_tgt.sentences), tokenizer_tgt.max_length)).astype(np.int64)

    for index, (data, label) in enumerate(zip(tokenizer_src.sentences, tokenizer_tgt.sentences)):
        source[index][:len(data)] = data
        target[index][:len(label)] = label

    return DataLoader(list(zip(source, target)), shuffle=True, batch_size=batch_size)