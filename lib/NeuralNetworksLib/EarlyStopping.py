import torch
import typing

class EarlyStopper:

    def __init__(self, patience : int, min_delta : float = 0.0, min_epoch : int = 200):
        self._patience = patience
        self._best_test_epoch = None
        self._best_test_loss = None
        self._min_delta = 1 - min_delta
        self._min_epoch = min_epoch

    def checkEpoch(self, testing_loss : float, epoch : int) -> typing.Tuple[bool, int]:
        if epoch < self._min_epoch:
            return True, 0

        if not self._best_test_loss or testing_loss < self._best_test_loss * self._min_delta:
            self._best_test_epoch = epoch
            self._best_test_loss = testing_loss

        epoch_delta = epoch - self._best_test_epoch
        if epoch_delta > self._patience:
            return False, epoch_delta

        return True, epoch_delta