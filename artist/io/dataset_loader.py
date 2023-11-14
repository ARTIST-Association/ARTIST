import pickle

class DataLoader():
    def __init__(self):
        pass

    def loadHelioFile(self, filename):
        file = open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        return data