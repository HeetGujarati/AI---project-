| Experiment | Method | Ratio (lambda) | Train Loss | Val Loss | Perplexity | Total Time (s) | Step Time (s) | Peak VRAM (MB) | Trainable Params |
|---|---|---|---|---|---|---|---|---|---|
| lora | LORA | 1x | 1.1289 | 1.0417 | 2.83 | 317.4 | 5.038 | 4037.6 | 9,232,384 (0.59%) |
| loraplus_4 | LORAPLUS | 4x | 1.1274 | 1.0377 | 2.82 | 388.2 | 6.163 | 4037.6 | 9,232,384 (0.59%) |
| loraplus_8 | LORAPLUS | 8x | 1.1297 | 1.0378 | 2.82 | 898.3 | 14.258 | 4037.6 | 9,232,384 (0.59%) |
| loraplus_16 | LORAPLUS | 16x | 1.1387 | 1.0424 | 2.84 | 232.7 | 3.694 | 4037.6 | 9,232,384 (0.59%) |
| loraplus_32 | LORAPLUS | 32x | 1.1641 | 1.0615 | 2.89 | 226.3 | 3.593 | 4037.6 | 9,232,384 (0.59%) |
| dyn_loraplus | DYN_LORAPLUS | Dyn(16→1) | 1.1180 | 1.0342 | 2.81 | 391.5 | 6.214 | 4037.6 | 9,232,384 (0.59%) |
