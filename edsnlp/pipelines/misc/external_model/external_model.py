"""`eds.dates` pipeline."""

from pathlib import Path
from typing import Iterable, Union

import dill
from spacy.language import Language
from spacy.tokens import Doc

from edsnlp.pipelines.base import BaseComponent

from .wrapper.wrapper import ModelWrapper


class ExternalModel(BaseComponent):
    """ """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        model: Union[str, Path, ModelWrapper],
        dataset_size: int,
        **kwargs,
    ):

        self.nlp = nlp

        if isinstance(model, ModelWrapper):
            self.model = model
        else:
            path = Path(model)
            if path.exists():
                with open(path, "rb") as f:
                    self.model = dill.load(f)
            else:
                raise ValueError(f"Could not find model at {str(path)}")

        self.model.prepare(**kwargs)
        self.dataset_size = dataset_size

    def pipe(
        self,
        stream: Iterable[Doc],
        batch_size: int,
    ):
        yield from self.model.predict(
            stream,
            dataset_size=self.dataset_size,
            batch_size=batch_size,
        )

    def __call__(self, doc: Doc):
        return next(
            self.model.predict(
                [doc],
                dataset_size=1,
                batch_size=1,
            )
        )
