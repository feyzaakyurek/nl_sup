class EvalResult:
    def __init__(self, values=None):
        if values is None:
            self.values = {}
            self.count = 0
        else:
            self.values = values
            self.count = 1

    def update(self, new_values):
        for k, v in new_values.values.items():
            if k not in self.values:
                self.values[k] = 0
            self.values[k] += v
        self.count += new_values.count

    def _norm_values(self):
        nv = {k: v/self.count for k, v in self.values.items()}
        nv["n"] = self.count
        return nv

    def __repr__(self):
        return f"EvalResult({self._norm_values()})"

