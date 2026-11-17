# batched_serving splits

research ID: E2a (version 2). Populations: the full development sets of the paper, OBQA 742 (`splits/development_and_calibration/obqa_dev_ids.json`, SHA-256 `80d1bbdd9148df3037401881533a2137b0d4ce9ee90a8ebb424fb413440bfbab`) and ARC 299 (`splits/development_and_calibration/arc_dev_ids.json`, SHA-256 `588d16b38e5db04b26a0d48feb5b4b96f4caa6af3bd5352626e59aa40063419c`), in the original development ID order.

- `batch_composition.csv`: for each model pair, dataset and batch size B (1, 8, 32), the consecutive IDs in each batch (the last batch keeps the remainder); the same batches are used by all three arms.
- `replay_order.csv`: the order in which the three arms ran each batch (`order_position` 0, 1, 2 within the batch); the arm order rotates left by (pair index + dataset index + batch-size index + batch index) mod 3 over [synchronous, early_return, reference]. Within each job the batches ran in the order B = 1, 8, 32 and, for each B, by batch index, so (B, batch_id, order_position) gives the execution order.
