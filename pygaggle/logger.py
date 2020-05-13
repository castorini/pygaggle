import coloredlogs


__all__ = []


coloredlogs.install(level='INFO',
                    fmt='%(asctime)s [%(levelname)s] %(module)s: %(message)s')
