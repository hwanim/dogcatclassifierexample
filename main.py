__metaclass__ = type

from microsoftbotframework import MsBot
from tasks import *
import os

from microsoftbotframework import ReplyToActivity

def make_response(message):
    input_image = #input message
    path = #extract the path of picture in input_image

    response_message = catdogclassifiation()
    if message["type"] == "message":
    ReplyToActivity(fill=message,
                    text=response_message).send()


bot = MsBot(port=int(os.environ['PORT']))
bot.add_process(make_response)
bot.run()
