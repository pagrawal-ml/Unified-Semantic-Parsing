""" vocabulary.py
    Contains functions for vocabulary building and
    processing.

    Reference for the additional vocabulary symbols.
    <PAD>  --  0
    <EOS>  --  1
    <OOV>  --  2
    <GO>   --  3
"""
from collections import Counter
import sys
import json


def load_vocab(word_to_id_file):
    f = open(word_to_id_file, 'r')
    word_to_id = eval(f.read())
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    return word_to_id, id_to_word


# def build_vocabulary(sent_path, top_k, max_sum_seq_len):
#     """ build a vocabulary index of words in the dataset.

#     Args:
#         sent_path : The path to file containing sentences to be indexed.
#         top_k     : Choose the top_k most frequent words to index the
#                     dictionary.
#         max_sum_seq_len : Max length of sentences to consider.

#     Returns:
#         word_to_id - Word to id mappings.
#     """
#     wordcount = Counter()
#     with open(sent_path) as sent_f:
#         sentences = sent_f.readlines()

#     for sentence in sentences:
#         tokens = sentence.split()
#         if len(tokens) > max_sum_seq_len:
#             tokens = tokens[:max_sum_seq_len]
#         wordcount.update(tokens)

#     print("Words in the vocabulary : %d" % len(wordcount))

#     count_pairs = wordcount.most_common()
#     count_pairs = wordcount.most_common(top_k - 4)
#     words, _ = list(zip(*count_pairs))
#     word_to_id = dict(zip(words, range(4, len(words) + 4)))

#     # word_to_id['<PAD>'] = 0
#     # word_to_id['<EOS>'] = 1
#     # word_to_id['<OOV>'] = 2
#     # word_to_id['<GO>'] = 3

#     return word_to_id

def build_vocabulary_simple(sent_path):
    """ build a vocabulary index of words in the dataset.

    Args:
        sent_path : The path to file containing sentences to be indexed.
        top_k     : Choose the top_k most frequent words to index the
                    dictionary.
        max_sum_seq_len : Max length of sentences to consider.

    Returns:
        word_to_id - Word to id mappings.
    """
    wordcount = Counter()
    with open(sent_path) as sent_f:
        sentences = sent_f.readlines()

    for sentence in sentences:
        tokens = sentence.split()
        #if len(tokens) > max_sum_seq_len:
        #    tokens = tokens[:max_sum_seq_len]
        wordcount.update(tokens)

    print("Words in the vocabulary : %d" % len(wordcount))

    count_pairs = wordcount.most_common()
    #count_pairs = wordcount.most_common(top_k - 4)
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(2, len(words) + 2)))

    # word_to_id['<PAD>'] = 0
    # word_to_id['<EOS>'] = 1
    # word_to_id['<OOV>'] = 2
    # word_to_id['<GO>'] = 3

    word_to_id['<GO>'] = 0
    word_to_id['<EOS>'] = 1
    return word_to_id


def tokens_to_ids(tok_list, word_to_id, tokens_per_field):
    seq_len = len(tok_list)

    tokens = [word_to_id.get(word, word_to_id['<OOV>']) for word in tok_list]

    if seq_len < tokens_per_field:
        tokens.extend([word_to_id.get('<PAD>')] * (tokens_per_field - seq_len))

    return seq_len, tokens

def create_vocab():
    src_vocab_file = sys.argv[1]
    tgt_vocab_file = sys.argv[2]
    print('src file ', src_vocab_file )
    src_vocab = build_vocabulary_simple(src_vocab_file)
    tgt_vocab = build_vocabulary_simple(tgt_vocab_file)
    print(src_vocab)
    print(tgt_vocab)
    src_file_out = open('src.vocab', 'w')
    tgt_file_out = open('tgt.vocab', 'w')
    src_file_out.write(json.dumps(src_vocab))
    tgt_file_out.write(json.dumps(tgt_vocab))

def create_vocab_geo():
    src_vocab_file = '../data/geoqueries_v2/train_q.txt'
    tgt_vocab_file = '../data/geoqueries_v2/train_f.txt'
    print('src file ', src_vocab_file )
    src_vocab = build_vocabulary_simple(src_vocab_file)
    tgt_vocab = build_vocabulary_simple(tgt_vocab_file)
    print(src_vocab)
    print(tgt_vocab)
    src_file_out = open('src_geo.vocab', 'w')
    tgt_file_out = open('tgt_geo.vocab', 'w')
    src_file_out.write(json.dumps(src_vocab))
    tgt_file_out.write(json.dumps(tgt_vocab))

def main():
    create_vocab()


if __name__ == '__main__':
    main()