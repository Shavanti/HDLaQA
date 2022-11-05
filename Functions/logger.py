import logging


class NoOP:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):

            pass
        return no_op

def get_logger(filename, rank, verbosity=1, name=None):
    if rank == 0:
        level_dict = {0: logging.DEBUG,  1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        fh = logging.FileHandler(filename, 'w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    else:
        logger = NoOP()

    return logger
