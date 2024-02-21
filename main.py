import os
import nltk
import torch
from timeit import default_timer as timer
from torchinfo import summary

from data.load_dataset import LoadDataset
from data_processing.processing_tools import limit_size_text_dataset, create_dataset_model, remove_duplicate
from model.seq2seq_transformer import Seq2SeqTransformer
from model.use_model import UseModel
from tokenizer.custom_tokenizer import CustomTokenizer

if __name__ == '__main__':
    nltk.download('punkt')

    PATH_TOKENIZER = 'resources/tokenizer/'
    BATCH_SIZE = 128

    loadDataset = LoadDataset()

    dataset_en, dataset_fr = loadDataset.importWMT14()

    print(len(dataset_en))
    start_time = timer()
    dataset_en, dataset_fr = limit_size_text_dataset(dataset_en, dataset_fr)
    print(len(dataset_en))
    dataset_en, dataset_fr = remove_duplicate(dataset_en, dataset_fr)
    print((f"Processing time = {(timer() - start_time):.3f}s | Number Token = {len(dataset_en)}"))

    # dataset_en = dataset_en[:20000]
    # dataset_fr = dataset_fr[:200000]

    tokenizer_en = CustomTokenizer(language='english')
    tokenizer_fr = CustomTokenizer(language='french')

    if not os.path.isfile(PATH_TOKENIZER + 'dataset-WMT14.en'):
        start_time = timer()
        tokenizer_en.tokenizer(dataset_en)
        print((f"Tokenizer time = {(timer() - start_time):.3f}s | Number Token = {len(tokenizer_en.word_to_index)}"))
        tokenizer_en.save_tokenizer('dataset-WMT14.en')
    tokenizer_en.load_tokenizer('dataset-WMT14.en')

    if not os.path.isfile(PATH_TOKENIZER + 'dataset-WMT14.fr'):
        start_time = timer()
        tokenizer_fr.tokenizer(dataset_fr)
        print((f"Tokenizer time = {(timer() - start_time):.3f}s | Number Token = {len(tokenizer_fr.word_to_index)}"))
        tokenizer_fr.save_tokenizer('dataset-WMT14.fr')
    tokenizer_fr.load_tokenizer('dataset-WMT14.fr')

    dataset = create_dataset_model(tokenizer_en, tokenizer_fr, BATCH_SIZE)

    SRC_VOCAB_SIZE = len(tokenizer_en.word_to_index)
    TGT_VOCAB_SIZE = len(tokenizer_fr.word_to_index)
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DROPOUT = 0.1
    NUM_EPOCHS = 10
    PAD_IDX = tokenizer_en.word_to_index['<pad>']
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NAME = 'Translator'
    print(DEVICE)

    transformer = Seq2SeqTransformer(pad_idx=PAD_IDX,
                                     num_encoder_layers=NUM_ENCODER_LAYERS,
                                     num_decoder_layers=NUM_DECODER_LAYERS,
                                     emb_size=EMB_SIZE,
                                     nhead=NHEAD,
                                     src_vocab_size=SRC_VOCAB_SIZE,
                                     tgt_vocab_size=TGT_VOCAB_SIZE,
                                     dim_feedforward=FFN_HID_DIM,
                                     dropout=DROPOUT)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(),
                                 lr=0.0001,
                                 betas=(0.9, 0.98),
                                 eps=1e-9)

    summary(transformer, depth=4)

    model = UseModel(transformer, NAME, DEVICE)
    model.train(optimizer, loss_fn, dataset, NUM_EPOCHS)
