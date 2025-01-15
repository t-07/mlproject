'''Any execution that happens we should be able to log everything - maintain the logs actually, eg if any
exception occurs, it should be logged'''

import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(), "Logs", LOG_FILE)
os.makedirs(logs_path,exist_ok=True) #means make directories of logs path even if there exist any just create more

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__=="__main__":
    logging.info("Logging has started")



