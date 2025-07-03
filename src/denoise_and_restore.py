from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import random
import tqdm
from config import args

def tokenize_and_delete(text, pct):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)

    delete_string = '<<<delete>>>'
    n_deletes = int(len(tokens) * pct)

    delete_indices = []
    for entity in named_entities:
        if hasattr(entity, 'label'):
            entity_type = entity.label()
            if entity_type.startswith('PERSON') or entity_type.startswith('LOCATION') or entity_type.startswith(
                    'ORGANIZATION'):
                for idx, (word, tag) in enumerate(tagged_tokens):
                    if word in entity.leaves():
                        delete_indices.append(idx)
                        if len(delete_indices) >= n_deletes:
                            break
                if len(delete_indices) >= n_deletes:
                    break


    exclude_tags = set(['PRP', 'PRP$', 'WP', 'WP$', 'EX', 'MD', 'CC', 'IN', 'DT', 'PDT', 'POS', 'TO', 'UH'])
    delete_indices = [idx for idx in delete_indices if tagged_tokens[idx][1] not in exclude_tags]


    while len(delete_indices) < n_deletes:
        random_idx = random.randint(0, len(tokens) - 1)
        if tagged_tokens[random_idx][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG',
                                            'VBN', 'VBP', 'VBZ', 'RB', 'RBR',
                                            'RBS'] and random_idx not in delete_indices:
            delete_indices.append(random_idx)

    # delete the selected indices
    for idx in delete_indices:
        tokens[idx] = delete_string


    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == delete_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1


    assert num_filled == n_deletes, f"num_filled {num_filled} != n_deletes {n_deletes}"
    text = ' '.join(tokens)
    return text




def count_deletes(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each deleted span with a sample from T5 delete_model
def replace_deletes(texts):
    n_expected = count_deletes(texts)
    #print('n_expected :' + str(n_expected) + '\n')
    stop_id = delete_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = delete_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = delete_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.delete_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return delete_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    #print(texts[0])
    # return the text in between each matched delete token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]
    #print(extracted_fills[0])
    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(deleted_texts, extracted_fills):
    # split deleted text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in deleted_texts]

    n_expected = count_deletes(deleted_texts)

    # replace each delete token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def denoise_texts_(texts, span_length, pct, ceil_pct=False):

    deleted_texts = [tokenize_and_delete(x, pct) for x in texts]
    #print("deleted_texts[0]:"+deleted_texts[0] + "**\n")
    raw_fills = replace_deletes(deleted_texts)
    #print("raw_fills[0]" + raw_fills[0] + "**\n")
    extracted_fills = extract_fills(raw_fills)
    #print("extracted_fills" + ' '.join(extracted_fills[0]) + "**\n")
    denoised_texts = apply_extracted_fills(deleted_texts, extracted_fills)
    #print("denoised_texts" + denoised_texts[0] + "**\n")
    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in denoised_texts:
        idxs = [idx for idx, x in enumerate(denoised_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        deleted_texts = [tokenize_and_delete(x, pct) for idx, x in enumerate(texts) if idx in idxs]
        #print(deleted_texts)
        raw_fills = replace_deletes(deleted_texts)
        #print(raw_fills)
        extracted_fills = extract_fills(raw_fills)
        new_denoised_texts = apply_extracted_fills(deleted_texts, extracted_fills)
        for idx, x in zip(idxs, new_denoised_texts):
            denoised_texts[idx] = x
        attempts += 1
    # tmp_lst = [tmp for tmp in denoised_texts if tmp != '']
    # denoised_texts = tmp_lst

    return denoised_texts


def denoise_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying denoises"):
        outputs.extend(denoise_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs