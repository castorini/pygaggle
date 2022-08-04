import coloredlogs
import logging


__all__ = []


coloredlogs.install(level='INFO',
                    fmt='%(asctime)s [%(levelname)s] %(module)s: %(message)s')
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
