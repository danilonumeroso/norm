from pathlib import Path
from typing import Any


def _dump_json(obj: Any, path: Path):
    import json

    try:
        json.dump(obj,
                  path.open("w"),
                  indent=2)
    except TypeError as ex:
        _dump_txt(obj, path)
        print(f"Unable to dump obj in json format because of the following exception: {ex}")
        print("Obj dumped as string")


def _load_json(path: Path) -> Any:
    import json

    return json.load(path.open("r"))


def _dump_bin(obj: Any, path: Path):
    import pickle as pkl

    pkl.dump(obj, path.open("wb"))


def _load_bin(path: Path) -> Any:
    import pickle as pkl

    return pkl.load(path.open("rb"))


def _dump_txt(obj: Any, path: Path):
    f = open(path, 'w')
    f.write(str(obj))


def _load_txt(path: Path) -> str:
    f = open(path, 'r')
    return str(f.read())


def _dump_torch(obj: Any, path: Path):
    from torch import save

    save(obj, path)


def _load_torch(path: Path) -> Any:
    raise NotImplementedError()


IO_HELPERS = {
    'json': (_dump_json, _load_json),
    'pkl': (_dump_bin, _load_bin),
    'pickle': (_dump_bin, _load_bin),
    'txt': (_dump_txt, _load_txt),
    'pth': (_dump_torch, _load_torch)
}
