'''
systems stuff:
- write to blockchain using gRPCs

api:
- POST/check_in()
- POST/upload_model()
- POST/send_global_model()

fed apis():
- GET/get_init_weights
- 
'''

'''
Workflow:
- send random init weights from central server to nodes
- pick any K nodes for participation.
- train model for E epochs ---------------------- update blockchain
- send to central server. it aggregates --------- update blockchain
- replace the global model
REPEAT FOR T ROUNDS
'''

