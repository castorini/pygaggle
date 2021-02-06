from pathlib import Path
import os

from pydantic import BaseSettings


class Settings(BaseSettings):
    # Cache settings
    cache_dir: Path = Path(os.getenv('XDG_CACHE_HOME',
                                     str(Path.home() / '.cache'))) / 'covidex'
    flush_cache: bool = False


class MsMarcoSettings(Settings):
    pass


class TRECCovidSettings(Settings):
    pass


class Cord19Settings(Settings):
    # T5 model settings
    t5_model_dir: str = 'gs://neuralresearcher_data/covid/data/model_exp304'
    t5_model_type: str = 't5-base'
