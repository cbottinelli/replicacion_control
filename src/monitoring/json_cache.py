
from cattrs.preconf.json import make_converter

# Imports agregados
from typing import Any, Union
from elk.func_correct.loaded_problem import ToEval

cache_dir = "../data"

json_converter = make_converter()

# Hook agregado por compatibilidad de versiones cattrs (a checkear)
json_converter.register_structure_hook(
    Union[ToEval, Any],
    lambda v, _: v,
)
