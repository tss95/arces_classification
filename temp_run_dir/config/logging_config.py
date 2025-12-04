import sys
import colorlog

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'customFormatter': {
            '()': 'colorlog.ColoredFormatter',
            'format': '%(log_color)sARCES - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s: %(message)s',
            'log_colors': {
                'DEBUG': 'blue',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'magenta',
            }
        }
    },
    'handlers': {
        'consoleHandler': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'customFormatter',
            'stream': 'ext://sys.stdout',
        }
    },
    'loggers': {
        'ARCES': {
            'level': 'ERROR',
            'handlers': ['consoleHandler'],
            'propagate': False
        }
    },
    'root': {
        'level': 'NOTSET',
        'handlers': []
    }
}
