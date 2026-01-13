# Hierarchical Federated Learning with Audit Ledger

This project implements a privacy-preserving, decentralized federated learning platform for healthcare analytics, designed for cross-hospital collaboration without raw data sharing.
Hospitals train local models on sensitive medical data (e.g., imaging). Models are shared only as weights or logits, never data. Regional servers participate in cyclic knowledge distillation (CKD) to exchange knowledge in a fully decentralized manner without a central aggregator or master node.
The system uses a lightweight, blockchain-inspired append-only audit ledger to:
record cryptographic hashes of models,
store signed metadata and training events,
provide tamper-evident traceability and accountability.
Each participating node maintains a Digital Twin that represents its current state (compute availability, model version, health status). Digital Twins are locally maintained and published as verifiable metadata, used only as constraints for participation—not for centralized scheduling or control.
The architecture emphasizes:
decentralization (no orchestration authority),
regulatory compliance (DPDP Act, HIPAA/GDPR-aligned design),
fault tolerance and auditability,
suitability for low-resource hospital environments.
The project blends distributed systems, federated learning, cybersecurity, and applied ML, focusing on system design correctness rather than model novelty.
