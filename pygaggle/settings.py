from pathlib import Path
import os

from pydantic import BaseSettings


class Settings(BaseSettings):
    # Cache settings
    cache_dir: Path = Path(os.getenv('XDG_CACHE_HOME',
                                     str(Path.home() / '.cache'))) / 'covidex'
    flush_cache: bool = False


class MsMarcoSettings(Settings):
    msmarco_index_path: str = 'data/index-msmarco-passage-20191117-0ed488'


class Cord19Settings(Settings):
    cord19_index_path: str = 'data/lucene-index-covid-paragraph'
    # T5 model settings
    t5_model_dir: str = 'gs://neuralresearcher_data/covid/data/model_exp377'
    # 'gs://neuralresearcher_data/covid/data/model_exp304'
    duo_t5_model_dir: str = 'gs://duo-t5/experiments/1/model_12000000'
    t5_model_type: str = 't5-base'
