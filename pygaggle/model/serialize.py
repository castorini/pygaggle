from dataclasses import dataclass
from pathlib import Path
import json
import logging
import os
import re

from tensorflow.python.lib.io import file_io
from transformers import T5Config, T5ForConditionalGeneration
import torch


__all__ = ['CachedT5ModelLoader']
TRANSFO_PREFIX = 'model.ckpt'


@dataclass
class CachedT5ModelLoader:
    url: str
    cache_path: Path
    cache_key: str
    model_type: str = 't5-base'
    flush_cache: bool = False

    def __post_init__(self):
        self.ckpt_url = os.path.join(self.url, 'checkpoint')
        self.model_cache_dir = self.cache_path / self.cache_key
        self.model_cache_dir.mkdir(exist_ok=True, parents=True)
        assert file_io.file_exists(self.ckpt_url), 'checkpoint file missing'
        self.ckpt_prefix = file_io.read_file_to_string(self.ckpt_url)

    def _fix_t5_model(self, model: T5ForConditionalGeneration):
        with torch.no_grad():  # Make more similar to TensorFlow implementation
            model.decoder.block[0].layer[1].EncDecAttention.\
                relative_attention_bias.weight.data.zero_()
        return model

    def load(self) -> T5ForConditionalGeneration:
        try:
            if not self.flush_cache:
                return self._fix_t5_model(
                    T5ForConditionalGeneration.from_pretrained(
                        str(self.model_cache_dir), from_tf=True,
                        force_download=False))
        except (RuntimeError, OSError):
            logging.info('T5 model weights not in cache.')
        m = re.search(r'model_checkpoint_path: "(.+?)"', self.ckpt_prefix)
        assert m is not None, 'checkpoint file malformed'

        # Copy over checkpoint data
        ckpt_patt = re.compile(rf'^{m.group(1)}\.(data-\d+-of-\d+|'
                               rf'index|meta)$')
        for name in file_io.list_directory(self.url):
            if not ckpt_patt.match(name):
                continue
            url = os.path.join(self.url, name)
            url_stat = file_io.stat(url)
            cache_file_path = self.model_cache_dir / ckpt_patt.sub(
                                rf'{TRANSFO_PREFIX}.\1', name)
            try:
                cs = os.stat(str(cache_file_path))
                if cs.st_size == url_stat.length and \
                   cs.st_mtime_ns > url_stat.mtime_nsec and \
                   not self.flush_cache:
                    logging.info(f'Skipping {name}...')
                    continue
            except FileNotFoundError:
                pass
            logging.info(f'Caching {name}...')
            file_io.copy(url, str(cache_file_path), overwrite=True)

        # Transformers expects a model config.json
        config = T5Config.from_pretrained(self.model_type)
        with open(str(self.model_cache_dir / 'config.json'), 'w') as f:
            json.dump(config.__dict__, f, indent=4)
        return self._fix_t5_model(T5ForConditionalGeneration.from_pretrained(
                        str(self.model_cache_dir), from_tf=True,
                        force_download=False))
