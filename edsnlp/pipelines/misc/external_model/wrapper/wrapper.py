from typing import Any, Dict, Iterable, Optional, Union

import dill
import spacy
from spacy.tokens import Doc, Span, Token
from spacy.util import minibatch

Assignable = Union[Doc, Span, Token]

DEFAULT_SPAN_GETTER = {
    "@span_getters": "spans-and-context",
    "n_before": 1,
    "n_after": 1,
    "return_type": "text",
    "mode": "sentence",
    "attr": "TEXT",
    "ignore_excluded": True,
    "with_ents": True,
    "with_spangroups": True,
    "output_keys": {"text": "text", "span": "span"},
}

DEFAULT_ANNOTATION_SETTER = {
    "@span_getters": "set-all",
}


class ModelWrapper:
    def __init__(
        self,
        model: Any,
        span_getter: Optional[Dict[str, Any]] = DEFAULT_SPAN_GETTER,
        annotation_setter: Optional[Dict[str, Any]] = DEFAULT_ANNOTATION_SETTER,
        **kwargs,
    ):
        self.model = model

        if span_getter is not None:
            self.span_getter = spacy.registry.resolve(dict(span_getter=span_getter))[
                "span_getter"
            ]
        if annotation_setter is not None:
            self.annotation_setter = spacy.registry.resolve(
                dict(annotation_setter=annotation_setter)
            )["annotation_setter"]

        self.prepare(**kwargs)

    def prepare(self, **kwargs):
        pass

    def predict(
        self,
        stream: Iterable[Doc],
        *,
        dataset_size: int = 1000,
        batch_size: int = 16,
    ):
        for outer_batch in minibatch(stream, dataset_size):
            data, spacy_data = self.span_getter(outer_batch)
            preds = self.model.predict(data, batch_size)
            self.annotation_setter(preds, spacy_data)
            yield from outer_batch

    def to_pickle(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)
