#! ./env/bin/python
from dotenv import load_dotenv
import mlflow
import os
from pathlib import Path
import sys


def main():
    # load environment variables, like tracking uri
    load_dotenv()
    # set USER_UID variable to pass to container
    os.environ['USER_UID']=str(os.getuid())
    # need absolute path for docker bind mount
    os.environ['DATA_PATH']=str(Path(os.environ['DATA_PATH']).absolute())
    params = {x.split('=')[0]: x.split('=')[1] for x in sys.argv[1:]} 
    mlflow.projects.run(uri='gnn-project', parameters=params)


if __name__ == "__main__":
    main()

