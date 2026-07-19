from typing import Dict, Optional, Type
from .base import BaseGenerator
from .sd_generator import SDGenerator
from .tileset_generator import TilesetGenerator
from .environment_generator import EnvironmentGenerator
from .prop_generator import PropGenerator
from .ui_generator import UIGenerator
from .ip_adapter_generator import IPAdapterGenerator


_generator_classes: Dict[str, Type[BaseGenerator]] = {
    "sd": SDGenerator,
    "tileset": TilesetGenerator,
    "environment": EnvironmentGenerator,
    "prop": PropGenerator,
    "ui": UIGenerator,
    "ip_adapter": IPAdapterGenerator,
}


def register_generator(name: str, generator_cls: Type[BaseGenerator]) -> None:
    _generator_classes[name] = generator_cls


def get_generator_class(name: str) -> Optional[Type[BaseGenerator]]:
    return _generator_classes.get(name)


def list_generators() -> Dict[str, Type[BaseGenerator]]:
    return dict(_generator_classes)


def create_generator(name: str, **kwargs) -> Optional[BaseGenerator]:
    cls = get_generator_class(name)
    if cls is None:
        return None
    return cls(**kwargs)
