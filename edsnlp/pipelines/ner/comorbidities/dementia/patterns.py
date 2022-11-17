main_pattern = dict(
    source="main",
    regex=[
        r"demence",
        r"dementiel",
        r"corps de le[vw]y",
        r"deficits?.chroniques?.cognitif",
        r"troubles?.mnesique?",
        r"troubles?.praxique",
        r"troubles?.attentionel",
        r"troubles?.degeneratifs? des fonctions sup",
        r"maladies?.cerebrales? degen",
        r"troubles? neurocogn",
        r"déficits?.cognitif",
        r"(trouble|dysfonction).{1,20} cogniti",
        r"atteinte.{1,7}spheres?cogniti",
        r"syndrome frontal",
        r"dysfonction.{1,25}cogni",
        r"(?<=d').+alzheimer",
        r"binswanger",
        r"gehring",
        r"\bpick",
        r"de guam",
        r"[kc]reutzfeld.{1,5}ja[ck]ob",
        r"huntington",
        r"korsako[fv]",
        r"atropie.{1,10}(cortico|hippocamp|cereb|lobe)",
    ],
)

acronym = dict(
    source="acronym",
    regex=[
        r"\bSLA\b",
        r"\bDFT\b",
        r"\bDFT",
        r"\bTNC\b",
    ],
    regex_attr="TEXT",
)

charcot = dict(
    source="charcot",
    regex=[
        r"charcot",
    ],
    exclude=dict(
        regex=[
            "pied de",
            "marie.?tooth",
        ],
        window=(-3, 3),
    ),
)


default_patterns = [
    main_pattern,
    acronym,
    charcot,
]
