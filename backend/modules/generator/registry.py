import inspect
from typing import Dict, Optional, Type
from .base import BaseGenerator
from .sd_generator import SDGenerator
from .tileset_generator import TilesetGenerator
from .environment_generator import EnvironmentGenerator
from .prop_generator import PropGenerator
from .ui_generator import UIGenerator
from .animation_generator import AnimationGenerator
from .portrait_generator import PortraitGenerator
from .ip_adapter_generator import IPAdapterGenerator


_generator_classes: Dict[str, Type[BaseGenerator]] = {
    "sd": SDGenerator,
    "tileset": TilesetGenerator,
    "environment": EnvironmentGenerator,
    "prop": PropGenerator,
    "ui": UIGenerator,
    "animation": AnimationGenerator,
    "portrait": PortraitGenerator,
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
    return _instantiate(cls, **kwargs)


def _requires_base_generator(cls: Type[BaseGenerator]) -> bool:
    """Return True if the generator class needs a ``base_generator`` argument."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    return "base_generator" in sig.parameters


def _instantiate(cls: Type[BaseGenerator], **kwargs) -> BaseGenerator:
    """Instantiate a generator class, defaulting the ``base_generator`` wrapper
    argument to an :class:`SDGenerator` when the caller did not supply one.

    Wrapper modules (tileset, environment, prop, ui, animation, portrait)
    decorate a base generator. The registry is a public factory for these
    modules, so constructing one without an explicit base should still work by
    falling back to the canonical SD 1.5 generator.
    """
    if _requires_base_generator(cls) and not kwargs.get("base_generator"):
        kwargs["base_generator"] = SDGenerator()
    return cls(**kwargs)
