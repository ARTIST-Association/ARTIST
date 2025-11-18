## Testing

``ARTIST`` uses ``pytest`` for running the test suite.

To run the tests make sure you have the project and all dependencies installed.
Refer to the [Installation](https://github.com/ARTIST-Association/ARTIST/blob/main/README.md#installation) instructions on how to install artist and make sure to choose the option of installing an editable version with developer dependencies: 
```bash
pip install -e ."[dev]"
```

All of our pytests lie in the ``tests/`` directory.

To run all tests execute:
```bash
pytest tests/
```

To run the tests with Coverage:
```bash
pytest --cov=artist tests/
```
