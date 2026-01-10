# Hierarchical Federated Learning with Audit Ledger

This project implements a **privacy-preserving federated learning system for healthcare** using a **hierarchical training architecture** and a **lightweight, blockchain-inspired audit ledger**.  
The design focuses on scalability, auditability, and regulatory compliance without the overhead of a full blockchain network.

---

## Overview

- **Data never leaves hospitals**
- **Model training is decentralized**
- **Aggregation is hierarchical (hospital → region → global)**
- **Auditability is ensured via an immutable ledger**
- **No on-chain model storage or computation**

---

## How It Works

### 1. Local Training (Hospital Level)
- Each hospital trains a local model on its **private patient data**
- Raw data is **never shared**
- Only model outputs (weights / logits) are used downstream

---

### 2. Regional Aggregation (Knowledge Distillation)
- Multiple hospitals report to a **regional server**
- Instead of averaging weights, the regional server:
  - Uses **knowledge distillation**
  - Trains a regional model by learning from hospital model outputs
- This helps handle **non-IID medical data** across hospitals

---

### 3. Global Model Training
- Multiple regional models are distilled into a **single global model**
- Global model captures cross-region medical patterns
- No raw data or hospital models are directly exposed

---

### 4. Audit Ledger (Blockchain-Inspired)
- After each regional and global update:
  - A **cryptographic hash** of the model is generated
  - Metadata (timestamp, round ID, contributor IDs) is recorded
- The ledger is:
  - **Append-only**
  - **Tamper-evident**
  - **Distributed across trusted nodes**

> Models are stored **off-chain** (e.g., object storage).  
> Only hashes and metadata are logged on the ledger.

---

## Key Ideas

- **No centralized training**
- **Hierarchical aggregation improves scalability**
- **Knowledge distillation handles data heterogeneity**
- **Models are stored off-chain**
- **Ledger provides auditability, not computation**
- **Blockchain is used as a trust layer, not an execution layer**

---

## Why This Design?

- Avoids the complexity and cost of full blockchains
- Scales better than fully peer-to-peer federated learning
- Meets healthcare compliance and audit requirements
- Practical to implement in real hospital networks

---

## What This Is *Not*

- ❌ Not a cryptocurrency
- ❌ Not smart-contract heavy
- ❌ Not on-chain ML training
- ❌ Not full decentralization at every layer

---

## Summary

This system combines **federated learning**, **hierarchical aggregation**, and a **lightweight audit ledger** to create a **practical, scalable, and compliant healthcare ML platform** that ba
