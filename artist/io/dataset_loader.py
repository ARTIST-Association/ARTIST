import pickle


class DataLoader:
    @staticmethod
    def load_helio_file(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
