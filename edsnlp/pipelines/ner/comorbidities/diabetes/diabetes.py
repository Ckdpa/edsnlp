"""`eds.comorbidities.diabetes` pipeline"""
from typing import Generator

from spacy.tokens import Doc, Span

from edsnlp.matchers.utils import get_text
from edsnlp.pipelines.ner.comorbidities.base import Comorbidity

from .patterns import default_patterns


class Diabetes(Comorbidity):
    def __init__(self, nlp, patterns):

        self.nlp = nlp
        if patterns is None:
            patterns = default_patterns

        super().__init__(
            nlp=nlp,
            name="diabetes",
            patterns=patterns,
        )

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        for span in spans:

            if span._.source == "complicated":
                span._.status = 2

            elif any([k.startswith("complicated") for k in span._.assigned.keys()]):
                span._.status = 2

            elif (
                get_text(span, "NORM", ignore_excluded=True) == "db"
            ) and not span._.assigned:
                # Huge chance of FP
                continue

            yield span