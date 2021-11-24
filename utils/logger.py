import logging


def getLogger(name, file_path):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(file_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger