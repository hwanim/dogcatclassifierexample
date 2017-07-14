__metaclass__ = type

from microsoftbotframework import MsBot
from servingapi import *
import os


bot = MsBot(port=int(os.environ['PORT']))
bot.add_process(make_response)
bot.run()
