from pathlib import Path
import os

from pydantic import BaseSettings


class Settings(BaseSettings):
    # T5 model settings
    t5_model_type: str = 't5-base'
    t5_max_length: int = 512

    # Cache settings
    cache_dir: Path = Path(os.getenv('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'covidex'
    flush_cache: bool = False


class MsMarcoSettings(Settings):
    msmarco_index_path: str = '/content/data/index-msmarco-passage-20191117-0ed488'

    # T5 model settings
    t5_model_dir: str = '/content/models/t5'

    # monoBERT model settings
    monobert_dir: str = '/content/models/monobert_msmarco_large'
    monobert_model_type: str = 'bert-large'




class Cord19Settings(Settings):
	cord19_index_path: str = 'data/lucene-index-covid-paragraph'

    # T5 model settings
    t5_model_dir: str = 'gs://neuralresearcher_data/covid/data/model_exp304'
