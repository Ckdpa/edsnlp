from functools import partial
from typing import Dict, List, Optional, Type, Union

import spacy
from pydantic import BaseModel, Extra, root_validator, validator
from spacy.tokens import Doc, Span, Token

from edsnlp.utils.extensions import rsetattr

Assignable = Union[Type[Doc], Type[Span], Type[Token]]


@spacy.registry.annotation_setters("set-all")
def configured_set_all():
    return set_all


def set_all(
    preds: List[Dict],
    spacy_data: List[Assignable],
):
    """
    Simple annotation setter.
    For a spaCy data (spacy_data[i]) and a prediction (preds[i]),
    will do in pseudo-code:
        spaCy_data._.k = v for k,v in prediction.items()

    Parameters
    ----------
    preds : List[Dict]
        List of predictions, output of the model
    spacy_data : List[Assignable]
        List of Doc, Token or Span from which the predictions were made
    """
    for pred, single_spacy_data in zip(preds, spacy_data):
        for k, v in pred.items():
            rsetattr(single_spacy_data, f"_.{k}", v)


@spacy.registry.annotation_setters("from-mapping")
def configured_assign_from_dict(
    mapping,
):
    mapping = AssignModel.parse_obj(mapping).dict()["__root__"]
    return partial(
        assign_from_dict,
        mapping=mapping,
    )


def assign_from_dict(
    preds: List[Dict],
    spacy_data: List[Assignable],
    mapping: Dict[str, Dict[str, Assignable]],
):
    """
    Simple annotation setter.
    For a spaCy data (spacy_data[i]) and a prediction (preds[i]),
    will do in pseudo-code:
        spaCy_data.v = prediction[k]  k,v in mapping.items()

    Parameters
    ----------
    preds : List[Dict]
        List of predictions, output of the model
    spacy_data : List[Assignable]
        List of Doc, Token or Span from which the predictions were made
    mapping : Dict[str, Dict[str, Assignable]]
        Dictionary were keys are keys of `preds`, and values are extensions to set
        on `spacy_data`
    """
    for pred, single_spacy_data in zip(preds, spacy_data):
        for k, attr in mapping.items():
            rsetattr(single_spacy_data, attr["attr"], pred[k])


class BaseAssignModel(BaseModel, arbitrary_types_allowed=True, extra=Extra.forbid):
    attr: str
    assigns: Optional[Assignable]

    @validator("assigns", pre=True)
    def assigns_validation(cls, v):
        if v is None:
            v = Span
        return v

    @root_validator
    def set_extension(cls, values):
        attr, assigns = values["attr"], values["assigns"]
        stripped_attr = attr.lstrip("_.")
        if not assigns.has_extension(stripped_attr):
            assigns.set_extension(stripped_attr, default=None)
        return values


class AssignModel(BaseModel, extra=Extra.forbid):

    __root__: Dict[str, Union[str, BaseAssignModel]]

    @validator("__root__", pre=True)
    def str_to_dict(cls, root):
        for k, v in root.items():
            if not isinstance(v, dict):
                root[k] = dict(attr=v, assigns=Span)
        return root
