import argparse
import json
import os
import re
import time
from multiprocessing.pool import ThreadPool
import nltk
import wikipedia
from allennlp.predictors import Predictor
from tqdm import tqdm
from unicodedata import normalize

def processed_line(method, line):
    nps, wiki_results, pages = method.exact_match(line)
    line['noun_phrases'] = nps
    line['predicted_pages'] = pages
    line['wiki_results'] = wiki_results
    return line

def process_line_with_progress(method, line, progress=None):
    if progress is not None and line['id'] in progress:
        return progress[line['id']]
    else:
        return processed_line(method, line)

class Doc_Retrieval:
    def __init__(self, database_path, add_claim=False, k_wiki_results=None):
        self.add_claim = add_claim
        self.k_wiki_results = k_wiki_results
        self.proter_stemm = nltk.PorterStemmer()
        self.tokenizer = nltk.word_tokenize
        self.predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

        self.db = {}
        for file in os.listdir(database_path):
            with open(os.path.join(database_path, file), 'r', encoding='utf-8') as f:
                for line in f:
                    line_json = json.loads(line.strip())
                    self.db[line_json['id']] = line_json['lines']

    def get_NP(self, tree, nps):
        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == "NP":
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] == "NP":
                    nps.append(tree['word'])
                    self.get_NP(tree['children'], nps)
                else:
                    self.get_NP(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                self.get_NP(sub_tree, nps)

        return nps

    def get_subjects(self, tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])
        return subjects

    def get_noun_phrases(self, line):
        claim = line['claim']
        tokens = self.predictor.predict(claim)
        nps = []
        tree = tokens['hierplane_tree']['root']
        noun_phrases = self.get_NP(tree, nps)
        subjects = self.get_subjects(tree)
        for subject in subjects:
            if len(subject) > 0:
                noun_phrases.append(subject)
        if self.add_claim:
            noun_phrases.append(claim)
        return list(set(noun_phrases))

    def get_doc_for_claim(self, noun_phrases):
        predicted_pages = []
        for np in noun_phrases:
            if len(np) > 300:
                continue
            i = 1
            while i < 12:
                try:
                    docs = wikipedia.search(np)
                    if self.k_wiki_results is not None:
                        predicted_pages.extend(docs[:self.k_wiki_results])
                    else:
                        predicted_pages.extend(docs)
                except (ConnectionResetError, ConnectionError, ConnectionAbortedError, ConnectionRefusedError):
                    print("Connection reset error received! Trial #" + str(i))
                    time.sleep(600 * i)
                    i += 1
                else:
                    break

        predicted_pages = set(predicted_pages)
        processed_pages = []
        for page in predicted_pages:
            page = page.replace(" ", "_")
            page = page.replace("(", "-LRB-")
            page = page.replace(")", "-RRB-")
            page = page.replace(":", "-COLON-")
            processed_pages.append(page)

        return processed_pages

    def np_conc(self, noun_phrases):
        noun_phrases = set(noun_phrases)
        predicted_pages = []
        for np in noun_phrases:
            page = np.replace('( ', '-LRB-')
            page = page.replace(' )', '-RRB-')
            page = page.replace(' - ', '-')
            page = page.replace(' :', '-COLON-')
            page = page.replace(' ,', ',')
            page = page.replace(" 's", "'s")
            page = page.replace(' ', '_')

            if len(page) < 1:
                continue
            doc_lines = self.db.get(normalize("NFD", page))
            if doc_lines is not None:
                predicted_pages.append(page)
        return predicted_pages

    def exact_match(self, line):
        noun_phrases = self.get_noun_phrases(line)
        wiki_results = self.get_doc_for_claim(noun_phrases)
        wiki_results = list(set(wiki_results))

        claim = normalize("NFD", line['claim'])
        claim = claim.replace(".", "")
        claim = claim.replace("-", " ")
        words = [self.proter_stemm.stem(word.lower()) for word in self.tokenizer(claim)]
        words = set(words)
        predicted_pages = self.np_conc(noun_phrases)

        for page in wiki_results:
            page = normalize("NFD", page)
            processed_page = re.sub("-LRB-.*?-RRB-", "", page)
            processed_page = re.sub("_", " ", processed_page)
            processed_page = re.sub("-COLON-", ":", processed_page)
            processed_page = processed_page.replace("-", " ")
            processed_page = processed_page.replace("–", " ")
            processed_page = processed_page.replace(".", "")
            page_words = [
                self.proter_stemm.stem(word.lower()) for word in self.tokenizer(processed_page) if len(word) > 0
            ]

            if all([item in words for item in page_words]):
                if ':' in page:
                    page = page.replace(":", "-COLON-")
                predicted_pages.append(page)
        predicted_pages = list(set(predicted_pages))
        return noun_phrases, wiki_results, predicted_pages

def get_map_function(parallel, p=None):
    assert not parallel or p is not None, "A ThreadPool object should be given if parallel is True"
    return p.imap_unordered if parallel else map

def main(db_file, k_wiki, in_file, out_file, add_claim=True, parallel=True):
    method = Doc_Retrieval(database_path=db_file, add_claim=add_claim, k_wiki_results=k_wiki)
    processed = dict()
    path = os.getcwd()
    lines = []
    with open(os.path.join(path, in_file), "r", encoding="utf-8") as f:
        for line in f.readlines():
            lines.append(json.loads(line.strip()))
    if os.path.isfile(os.path.join(path, in_file + ".progress")):
        with open(os.path.join(path, in_file + ".progress"), 'rb') as f_progress:
            import pickle
            progress = pickle.load(f_progress)
            print(os.path.join(path, in_file + ".progress") + " exists. Load it as progress file.")
    else:
        progress = dict()

    try:
        with ThreadPool(processes=4 if parallel else None) as p:
            for line in tqdm(get_map_function(parallel, p)(lambda l: process_line_with_progress(method, l, progress),
                                                           lines),
                             total=len(lines)):
                processed[line['id']] = line
                progress[line['id']] = line
        with open(os.path.join(path, out_file), "w+") as f2:
            for line in lines:
                f2.write(json.dumps(processed[line['id']]) + "\n")
    finally:
        with open(os.path.join(path, in_file + ".progress"), 'wb') as f_progress:
            import pickle
            pickle.dump(progress, f_progress, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-file', type=str, help="database file which saves pages")
    parser.add_argument('--in-file', type=str, help="input dataset")
    parser.add_argument('--out-file', type=str, help="path to save output dataset")
    parser.add_argument('--k-wiki', type=int, help="first k pages for wiki search")
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--add-claim', type=bool, default=True)
    args = parser.parse_args()

    main(args.db_file, args.k_wiki, args.in_file, args.out_file, args.add_claim, args.parallel)
