from dataclasses import dataclass, asdict
import json

@dataclass
class BaseConfiguration:
    def __init__(self):
        raise NotImplementedError("Please use one of the extended classes.")
    
    def to_dict(self):
        """converts configurations elements into a dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, lc_dict):
        """creates a `BaseConfiguration` instance from a dict."""
        return cls(**lc_dict)

    def to_json(self, file_path):
        """write out the `BaseConfiguration` as a json file."""
        with open(file_path, 'w') as fp:
            json.dump(self.to_dict(), fp)

    @classmethod
    def from_json(cls, file_path):
        """read a `BaseConfiguration` generated json file and instantiate."""
        with open(file_path) as fp:
            lc_dict = json.load(fp)
            return cls(**lc_dict)