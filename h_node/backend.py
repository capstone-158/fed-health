'''
systems stuff:
- write to blockchain using gRPCs

api:
- check_in() POST
- upload_model() POST
- send_global_model() GET

'''

from fastapi import FastAPI

app = FastAPI()

node_registry = {}

