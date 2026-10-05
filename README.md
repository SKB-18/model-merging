# Model Merging & Distillation

This is Repo 5 (The Stretch Goal) of the AI Project Portfolio Roadmap.

It implements techniques for combining models to get the best of both worlds without pretraining from scratch.

## Architecture
- **`merge/slerp.py`**: Implementation of Spherical Linear Interpolation (SLERP) for interpolating model weight vectors. SLERP is highly effective for language models because it maintains the variance and geometric properties of the high-dimensional weight matrices better than simple linear interpolation.

## Testing
```bash
python -m pytest tests/
```
