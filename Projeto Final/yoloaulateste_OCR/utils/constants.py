import os
import environ
from os.path import join, dirname, abspath


#env = environ.Env()
env_path_file = join(dirname(dirname(abspath(__file__))), '.env')

#env.read_env(env_path_file)

SMTP_SERVER_ADDRESS = os.environ.get('SMTP_SERVER_ADDRESS')
PORT = os.environ.get('PORT')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD')
SENDER_ADDRESS = os.environ.get('SENDER_ADDRESS')
