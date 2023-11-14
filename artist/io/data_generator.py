import pickle

class DataGenerator():
    def __init__(self):
        pass

    def generate(self, filename, targetHeliostat, lightSource):
        data = {
            "targetHeliostat": targetHeliostat,
            "lightSource": lightSource
        }
        file = open(filename, 'wb')
        pickle.dump(data, file)
        file.close()