import load_datasets
from config import args
import numpy as np
import transformers

# load steganalysis text data
def generate_stega_data(dataset, key):
    data_cover, data_stego = load_datasets.load_xsum(args.datadir)

    data_cover = stegatext_process(data_cover)
    data_stego = stegatext_process(data_stego)

    final_data = {
        "cover": data_cover,
        "stego": data_stego,
    }
    return final_data


def stegatext_process(data):
    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()
    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    # if dataset in ['Imdb', 'tweet']: modification:delete if condition
    # modification >150
    long_data = [x for x in data if len(x.split()) > 150 and len(x.split()) < 300]
    if len(long_data) > 0:
        data = long_data
    # keep only examples with <= 512 tokens according to delete_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")
    return data

def strip_newlines(text):
    return ' '.join(text.split())