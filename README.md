# Hierarchical Federated Learning with Audit Ledger

This project implements a privacy-preserving, decentralized federated learning platform for healthcare analytics, designed for cross-hospital collaboration without raw data sharing.

---

Hospitals train local models on sensitive medical data (e.g., imaging). Models are shared only as weights or logits, never data. Regional servers participate in cyclic knowledge distillation (CKD) to exchange knowledge in a fully decentralized manner without a central aggregator or master node.

---

# High level working

Training Round:
1. Nodes train locally → send weights to regional server
2. Regional server aggregates → creates regional model
3. Regional servers perform CKD with EACH OTHER (not blockchain)
4. Regional servers compute global model
5. Regional servers log hashes to blockchain (audit only)
6. Regional servers send global model to nodes

---

# How to run
Prerequisites:
```bash
brew install --cask orbstack
brew install docker

# In the main project directory
mkdir data/ models/
git pull origin main
```

for macos:
```bash
orb start
docker compose up --build
```

todo: check TODO markers