import argparse




class getOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Parses command.")
        self.parser.add_argument("-s", "--shots", type=int, help="number of shots to be used")
        self.parser.add_argument("-tm", "--testing_model", type=str, help="model weights for testing")
        self.parser.add_argument("-c", "--cipher", type=str, help="Your cipher name")
        self.parser.add_argument("-tr", "--thresh", type=float, help="threshold of reading confidence")
        self.parser.add_argument("-ap", "--alphabet", type=str, help="alphabet path")
        self.parser.add_argument("-lp", "--lines", type=str, help="lines path")
        self.parser.add_argument("-op", "--output", type=str, help="output path")
    def parse(self):
        return self.parser.parse_args()
    