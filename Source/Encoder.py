from sklearn.decomposition import PCA
import pandas

class VecEncoder:
    def __init__(self, maxDims = 3) -> None:
        self.m_maxDims = maxDims
        self.m_delimiter = "|"

    def simplify(self, source: pandas.DataFrame, name: str) -> pandas.DataFrame:
        print(f"Reducing dims: {len(source.columns)} -> {self.m_maxDims}")
        source = source - source.mean()
        source = source / source.var()
        evaluator = PCA(n_components = self.m_maxDims)
        result = evaluator.fit_transform(source)
        result = pandas.DataFrame(result)
        result.columns = [f"{name}|PC{i}" for i in result]
        return result

    def encode(self, source: pandas.Series) -> pandas.DataFrame:
        indices = list(range(len(source)))
        prefix = source.name + self.m_delimiter
        columns = [prefix + ID for ID in source.unique()]
        newFrame = pandas.DataFrame(index = indices, columns = columns)
        for val in source.unique():
            rowIdx = source == val
            colIdx = source.name + self.m_delimiter + val
            newFrame.loc[rowIdx] = 0.0
            newFrame.loc[rowIdx, colIdx] = 1.0
        newFrame = newFrame.astype(pandas.UInt8Dtype()) # NOTE: Size reduction
        return newFrame

    def __call__(self, source: pandas.DataFrame, column: str) -> pandas.DataFrame:
        result = self.encode(source[column])
        if(len(result.columns) > self.m_maxDims):
            result = self.simplify(result, column)
        source = source.drop(columns = column)
        source = pandas.concat([source, result], axis = 1)
        return source

class LabelEncoder:
    def __init__(self, bindings: dict | None = None) -> None:
        self.m_bindings = bindings

    def createBindigs(self, source: pandas.Series) -> dict:
        keys = source.unique()
        result = {key : idx + 1 for idx, key in enumerate(keys)}
        return result

    def __call__(self, source: pandas.DataFrame, column: str) -> pandas.DataFrame:
        if(self.m_bindings is None):
            self.m_bindings = self.createBindigs(source[column])
        indices = list(range(source.shape[0]))
        result = source[column].replace(self.m_bindings)
        result = pandas.to_numeric(result, errors = "coerce")
        source[column] = result
        return source

class FreqEncoder:
    def createBindigs(self, source: pandas.Series) -> dict:
        result = {}
        for val in source.unique():
            mask = source == val
            count = mask.sum().item()
            result[val] = count / len(source)
        return result

    def __call__(self, source: pandas.DataFrame, column: str) -> pandas.DataFrame:
        bindings = self.createBindigs(source[column])
        indices = list(range(source.shape[0]))
        result = source[column].replace(bindings)
        source[column] = result
        return source
