__metaclass__ = type

from microsoftbotframework import MsBot
from tasks import *
import os

from microsoftbotframework import ReplyToActivity

bot = MsBot(port=int(os.environ['PORT']))
bot.add_process(make_response)
bot.run()
