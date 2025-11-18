## Testing

``ARTIST`` uses ``pytest`` for running the test suite.
To run the tests make sure you have the project and all dependencies installed.
All of our pytests lie in the ``tests/`` directory.
To run all tests execute:
```bash
pytest tests/
```

To run the tests with Coverage:
```bash
pytest --cov=artist tests/