import random
import datasets


DATASETS = ['imdb', 'Tweet', 'news', 'writing_prompts', 'xsum']
def load_Imdb(datapath, cache_dir=None):
    imdb_path_cover = 'data/imdb/imdb_cover'
    imdb_path_stego = datapath
    with open(f'{imdb_path_cover}.txt', 'r', encoding='utf-8') as fc:
        reviews_c = fc.readlines()
    filtered_c = [process_Imdb(review) for review in reviews_c]
    with open(f'{imdb_path_stego}.txt', 'r', encoding='utf-8') as fs:
        reviews_s = fs.readlines()
    filtered_s = [process_Imdb(review) for review in reviews_s]
    random.seed(0)
    random.shuffle(filtered_c)
    random.shuffle(filtered_s)
    # with open(f'{imdb_path}1.txt', 'w', encoding='utf-8') as f:
    #     passages = [filter + '\n' for filter in filtered]
    #     f.writelines(passages)
    return filtered_c, filtered_s

def load_writing_prompts(datapath, cache_dir=None):
    imdb_path_cover = 'data/writing_prompts/writing_prompts_cover'
    imdb_path_stego = datapath
    with open(f'{imdb_path_cover}.txt', 'r', encoding='utf-8') as fc:
        reviews_c = fc.readlines()
    filtered_c = [process_Imdb(review) for review in reviews_c]
    with open(f'{imdb_path_stego}.txt', 'r', encoding='utf-8') as fs:
        reviews_s = fs.readlines()
    filtered_s = [process_Imdb(review) for review in reviews_s]
    random.seed(0)
    random.shuffle(filtered_c)
    random.shuffle(filtered_s)
    # with open(f'{imdb_path}1.txt', 'w', encoding='utf-8') as f:
    #     passages = [filter + '\n' for filter in filtered]
    #     f.writelines(passages)
    return filtered_c, filtered_s

def load_xsum(datapath, cache_dir=None):
    imdb_path_cover = 'data/xsum/xsum_cover'
    imdb_path_stego = datapath
    with open(f'{imdb_path_cover}.txt', 'r', encoding='utf-8') as fc:
        reviews_c = fc.readlines()
    filtered_c = [process_Imdb(review) for review in reviews_c[:200]]
    with open(f'{imdb_path_stego}.txt', 'r', encoding='utf-8') as fs:
        reviews_s = fs.readlines()
    filtered_s = [process_Imdb(review) for review in reviews_s]
    random.seed(0)
    random.shuffle(filtered_c)
    random.shuffle(filtered_s)
    # with open(f'{imdb_path}1.txt', 'w', encoding='utf-8') as f:
    #     passages = [filter + '\n' for filter in filtered]
    #     f.writelines(passages)
    return filtered_c, filtered_s

def load_news(cache_dir=None):
    imdb_path_cover = 'data/news/ac/cover'
    imdb_path_stego = 'data/news/ac/stego'
    with open(f'{imdb_path_cover}.txt', 'r', encoding='utf-8') as fc:
        news_c = fc.readlines()
    filtered_c = [process_news(review) for review in news_c]
    length = len(filtered_c)
    tmpc = []
    for i in range(length // 10):
        tmpc.append('. '.join(filtered_c[i : i+10]))
    processed_tmp = [process_Imdb(news) for news in tmpc]
    filtered_c = processed_tmp
    with open(f'{imdb_path_stego}.txt', 'r', encoding='utf-8') as fs:
        news_s = fs.readlines()
    filtered_s = [process_news(review) for review in news_s]
    length = len(filtered_s)
    tmps = []
    for i in range(length // 10):
        tmps.append('. '.join(filtered_s[i: i + 10]))
    processed_tmp = [process_Imdb(news) for news in tmps]
    filtered_s = processed_tmp
    random.seed(0)
    random.shuffle(filtered_c)
    random.shuffle(filtered_s)
    # with open(f'{imdb_path_cover}1.txt', 'w', encoding='utf-8') as f:
    #     passages = [filter + '\n' for filter in filtered_c]
    #     f.writelines(passages)
    return filtered_c, filtered_s

def process_Imdb(review):
    return review.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').replace(
        '<br /><br />', '\t').strip()
def process_news(news):
    return news.capitalize().replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').replace(
        '<br /><br />', '\t').strip()


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()



def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')

