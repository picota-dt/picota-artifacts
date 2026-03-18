import json


class DatasetLoader:
    def __init__(self, path):
        self.lookback = None
        self.input_variables = None
        self.stds = None
        self.means = None
        self.out_min = None
        self.out_max = None
        self.path = path

    def load(self):
        data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            first_line = next(f).strip()
            if not first_line:
                raise ValueError("The file is empty or means and stds are missing.")
            stats = json.loads(first_line)
            self.means = stats.get("means")
            self.stds = stats.get("stds")
            self.out_min = stats.get("out_min")
            self.out_max = stats.get("out_max")
            self.lookback = stats.get("lookback_size")
            self.input_variables = stats.get("input_variables")
            if self.means is None or self.stds is None:
                raise KeyError("First line should contain 'means' y 'stds'.")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                data.append(obj)
        return data

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    def get_out_min(self):
        return self.out_min

    def get_out_max(self):
        return self.out_max

    def get_input_variables(self):
        return self.input_variables

    def get_lookback(self):
        return self.lookback