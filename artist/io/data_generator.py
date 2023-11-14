import pickle


class DataGenerator:
    def __init__(self):
        pass

    def generate(self, filename, target_heliostat, light_source):
        data = {"targetHeliostat": target_heliostat, "lightSource": light_source}
        with open(filename, "wb") as file:
            pickle.dump(data, file)
