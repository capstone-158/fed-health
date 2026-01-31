# get node id and assign to a region using some maths
import os

'''
apis:
- upload_local_model

- check_in() upon stddartup

'''
'''
systems part:
- send model to reg server (gRPCs)
- heartbeats
'''

from fastapi import FastAPI
import logging

logger = logging.basicConfig()

app = FastAPI()

def check_in():
    node_id = os.getenv()