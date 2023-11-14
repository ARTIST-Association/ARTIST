import pickle


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_helio_file(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
