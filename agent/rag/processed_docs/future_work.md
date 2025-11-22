# Future Work

## Research-Oriented Extensions
1. **Multi-agent Trajectory Prediction** – Jointly reason about multiple nearby vehicles, incorporating interaction-aware models or graph-based neural networks.
2. **Transformer or Diffusion Models** – Replace the LSTM with attention-based architectures or probabilistic trajectory samplers for richer uncertainty estimates.
3. **Sensor Fusion** – Integrate LiDAR point clouds, camera detections, V2X broadcasts, or high-definition maps to enrich context beyond scalar telemetry.
4. **Counterfactual Explanations** – Extend the explanation tool to highlight how different control inputs or lane decisions would alter the predicted trajectory or risk.

## Product & Deployment Enhancements
1. **CARLA / SUMO Integration** – Couple the system with simulators to create closed-loop evaluation and reinforcement-learning scenarios.
2. **Edge Deployment** – Optimize the model via pruning, quantization, or TensorRT so it can run on automotive-grade hardware.
3. **Continuous Learning Pipeline** – Automate retraining with fresh fleet data, handling drift detection and evaluation dashboards.
4. **Compliance & Audit Trails** – Generate signed logs of predictions, explanations, and RAG sources for regulatory audits.
5. **User Management & Authentication** – Secure the FastAPI endpoints with API keys or OAuth for enterprise deployment.
