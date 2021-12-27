import logging
import datetime
import yaml


class Logger(object):
    def __init__(self, filename, pure=True):
        self.filename = filename
        self.pure = pure
        self.logger = logging.getLogger()
        self.init()

    def init(self):
        self.logger.setLevel('INFO')
        # LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        LOG_FORMAT = "%(message)s" if self.pure else "%(asctime)s | %(message)s"
        DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        screenHandler = logging.StreamHandler()
        screenHandler.setFormatter(formatter)
        fileHandler = logging.FileHandler(self.filename)
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(screenHandler)
        self.logger.addHandler(fileHandler)
        # logging.basicConfig(filename=self.filename, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    def log(self, msg):
        # logging.info(msg)
        self.logger.info(msg)


def get_nowtime():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def load_yaml(path):
    with open(path ,'r', encoding='utf-8') as f:
        # return yaml.load(f.read(),Loader=yaml.FullLoader)
        return yaml.load(f.read())


if __name__ == '__main__':
    logger = Logger('my.log')
    logger.log('This is a log.')
