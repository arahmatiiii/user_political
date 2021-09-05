from transformers import T5EncoderModel, T5Tokenizer
import glob
import pandas
import hazm


def find_num_of_upper(seq_lens, upper):
    num_of_upper = 0
    for item in seq_lens:
        if item >= upper:
            num_of_upper += 1
    return num_of_upper


if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained("/mnt/ali.rahmati/work_th/assets/mt5_en_large")

    max_len_text_tokenizer = []
    max_len_text_split = []

    dataset = pandas.read_csv('/mnt/ali.rahmati/work_th/data/Processed/train_data_normed.csv')

    for i in range(len(dataset)):
        pre_len = len(tokenizer.tokenize(str(dataset['text'][i])))
        max_len_text_tokenizer.append(pre_len)
        max_len_text_split.append(len(str(dataset['text'][i]).split()))

    print(len(max_len_text_tokenizer), find_num_of_upper(max_len_text_tokenizer, 120))
    print(max(max_len_text_tokenizer))
    print(len(max_len_text_split), find_num_of_upper(max_len_text_split, 80))
    print(max(max_len_text_split))
