import os
import shutil
import tarfile
import unittest
from random import randint
from typing import List
from urllib.request import urlretrieve

from pyserini.pyclass import JSimpleSearcherResult
from pyserini.search import pysearch

from pygaggle.rerank import to_texts, Text, Query, Reranker
from pygaggle.rerank import IdentityReranker


class TestSearch(unittest.TestCase):
    def setUp(self):
        # Download pre-built CACM index
        # Append a random value to avoid filename clashes.
        r = randint(0, 10000000)
        self.collection_url = ('https://github.com/castorini/anserini-data/'
                               'raw/master/CACM/lucene-index.cacm.tar.gz')
        self.tarball_name = 'lucene-index.cacm-{}.tar.gz'.format(r)
        self.index_dir = 'index{}/'.format(r)

        _, _ = urlretrieve(self.collection_url, self.tarball_name)

        tarball = tarfile.open(self.tarball_name)
        tarball.extractall(self.index_dir)
        tarball.close()

        self.searcher = pysearch.SimpleSearcher(
            f'{self.index_dir}lucene-index.cacm')

    def test_basic(self):
        hits = self.searcher.search('information retrieval')

        self.assertTrue(isinstance(hits, List))

        self.assertTrue(isinstance(hits[0], JSimpleSearcherResult))
        self.assertEqual('CACM-3134', hits[0].docid)
        self.assertEqual(3133, hits[0].lucene_docid)
        self.assertEqual(1500, len(hits[0].contents))
        self.assertEqual(1532, len(hits[0].raw))
        self.assertAlmostEqual(4.76550, hits[0].score, places=5)

        texts = to_texts(hits)
        self.assertEqual(len(hits), len(texts))
        self.assertTrue(isinstance(texts, List))
        self.assertTrue(isinstance(texts[0], Text))

        for i in range(0, len(hits)):
            self.assertEqual(hits[i].raw, texts[i].raw)
            self.assertEqual(hits[i].contents, texts[i].contents)
            self.assertAlmostEqual(hits[i].score, texts[i].score, places=5)

        query = Query('dummy query')
        identity_reranker = IdentityReranker()
        self.assertTrue(isinstance(identity_reranker, Reranker))

        output = identity_reranker.rerank(query, texts)

        # Check that reranked output is indeed the same as the input
        for i in range(0, len(hits)):
            self.assertEqual(texts[i].raw, output[i].raw)
            self.assertEqual(texts[i].contents, hits[i].contents)
            self.assertAlmostEqual(texts[i].score, hits[i].score, places=5)

        # Check that the identity rerank was not destructive
        texts = []
        for i in range(0, len(hits)):
            self.assertEqual(hits[i].raw, output[i].raw)
            self.assertEqual(hits[i].contents, output[i].contents)
            self.assertAlmostEqual(hits[i].score, output[i].score, places=5)

    def tearDown(self):
        self.searcher.close()
        os.remove(self.tarball_name)
        shutil.rmtree(self.index_dir)


if __name__ == '__main__':
    unittest.main()
