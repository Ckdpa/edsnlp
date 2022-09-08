from typing import Any, List, Optional, Union, Dict

from spacy.language import Language

from .cerebrovascular_accident import CerebrovascularAccident

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.comorbidities.cerebrovascular_accident",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return CerebrovascularAccident(nlp, patterns=patterns)
