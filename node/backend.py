# get node id and assign to a region using some maths

'''
systems part:
- send model to reg server (gRPCs)
- heartbeats
'''

from fastapi import FastAPI
import logging

logger = logging.basicConfig()

app = FastAPI()

'''
apis:
- upload_local_model

'''