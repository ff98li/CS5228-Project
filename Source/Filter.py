import pandas

class CorrFilter:
    def __init__(self, threshold: float = 0.95) -> None:
        self.m_threshold = threshold

    def eliminate(self, source: pandas.DataFrame, score) -> pandas.DataFrame | None:
        for idx, value in enumerate(score):
            if(value > 1):
                label = source.columns[idx]
                print(f"Dropped column: {label}")
                source = source.drop(labels = label, axis = 1)
                return source
        return None

    def __call__(self, source: pandas.DataFrame) -> pandas.DataFrame:
        while(True):
            values = abs(source.corr())
            values = values > self.m_threshold
            values = values.sum(axis = 0)
            result = self.eliminate(source, values)
            if(result is None):
                break
            source = result
        return source
