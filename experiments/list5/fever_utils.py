import re
import spacy
from transformers import T5Tokenizer

# regex patterns in order of priority
PTB_SYMBOLS = [
    (re.compile(r'-LRB-'), '('),
    (re.compile(r'-RRB-'), ')'),
    (re.compile(r'\( ([^\(\)]*?) \)'), '(\\1)'),
    (re.compile(r'-LSB-'), '['),
    (re.compile(r'-RSB-'), ']'),
    (re.compile(r'\[ ([^\[\]]*?) \]'), ''),  # most occurrences of [] contain pronounciations, which we don't want
    (re.compile(r'-LCB-'), '{'),
    (re.compile(r'-RCB-'), '}'),
    (re.compile(r'\{ ([^\{\}]*?) \}'), '{\\1}'),
    (re.compile(r'-COLON-'), ':'),
    (re.compile(r'–|−'), '-'),
    (re.compile(r'`` ([^`]*?) \'\''), '"\\1"'),
    (re.compile(r'` ([^`]*?) \''), '\'\\1\''),
    (re.compile(r' ([,\.:;\'!?])'), '\\1')
]

# NER
nlp = spacy.load('en_core_web_sm')

# T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-3b')

def extract_sentences(lines):
    """
    Extracts the non-empty sentences and their numbers of the "lines" field in
    a JSON object from a FEVER wiki-pages JSONL file.
    """
    sentences = []

    sentence_index = 0
    for line in lines.split('\n'):
        tokens = line.split('\t')
        if not tokens[0].isnumeric() or int(tokens[0]) != sentence_index:
            # skip non-sentences, caused by unexpected \n's
            continue
        else:
            sentences.append((tokens[1], tokens[0]))
            sentence_index += 1

    return sentences

def make_sentence_id(doc_id, sentence_num):
    """
    Returns the sentence ID of a Wikipedia document ID and the number
    corresponding to its specific sentence index in the document.
    """
    return f'{doc_id}_{sentence_num}'

def split_sentence_id(sentence_id):
    """
    Returns the original document ID and sentence number of a sentence ID.
    """
    separator_index = sentence_id.rfind('_')
    doc_id = sentence_id[:separator_index]
    sent_num = int(sentence_id[separator_index + 1:])

    return doc_id, sent_num

def normalize_text(text):
    """
    Normalizes text found in FEVER dataset, removing punctuation tokens and
    cleaning whitespace around punctuation.
    """
    for regexp, substitution in PTB_SYMBOLS:
        text = regexp.sub(substitution, text)

    return text

def remove_disambiguation(doc_id):
    """
    Normalizes and removes disambiguation info from a document ID.
    """
    doc_id = doc_id.replace('_', ' ').replace('-COLON-', ':')
    if '-LRB-' in doc_id:
        doc_id = doc_id[:doc_id.find('-LRB-') - 1]

    return doc_id

def extract_entities(text):
    """
    Extracts named entities from text using spaCy's en_core_web_sm NER module.
    """
    doc = nlp(text)
    ner_entities = list(
        set([
            entity.text for entity in doc.ents
            # ignore entities that are less likely to correspond to a Wikipedia article
            if entity.label_ not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
        ]))

    return ner_entities

def truncate(query, sent, num_sents, line_len, trunc_count):
    """
    Truncates evidence sentence to fit in a T5 input line with max line_len
    tokens and num_sents sentences to be concatenated. Accounts for query text
    and tracks the number of sentences truncated.
    """
    query_tokens = tokenizer.tokenize(query)
    trunc_len = (line_len - 2 - len(query_tokens) - 3 * num_sents - 1) // num_sents
    tokens = tokenizer.tokenize(sent)
    if len(tokens) > trunc_len:
        tokens = tokens[:trunc_len]
        return tokenizer.convert_tokens_to_string(tokens), trunc_count + 1

    return sent, trunc_count
