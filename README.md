# Calibrated Residual-Stream Interventions Reveal Partial but Not Strong Train--Test Separation in Modular-Addition Grokking

Code for the calibrated separability experiments reported in the paper.

**Paper:** [arXiv link forthcoming]

## Setup

```bash
python3 -m venv modular-addition-env
modular-addition-env/bin/python -m pip install -r requirements.txt
```

## Tests

```bash
modular-addition-env/bin/python -m unittest discover -s tests -p 'test_calibrated_separability.py' -v
```

## License

MIT
