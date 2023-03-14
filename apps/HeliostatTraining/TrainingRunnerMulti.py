from threading import Thread
import TrainingRunner
import time

for i in range(5):
    Thread(target=TrainingRunner.main).start()
    time.sleep(10)