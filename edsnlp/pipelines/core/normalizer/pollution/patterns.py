# noinspection SpellCheckingInspection
information = [
    (
        r"(?s)(=====+\s*)?(L\s*e\s*s\sdonnées\s*administratives,\s*sociales\s*|"
        r"I?nfo\s*rmation\s*aux?\s*patients?|"
        r"L[’']AP-HP\s*collecte\s*vos\s*données\s*administratives|"
        r"L[’']Assistance\s*Publique\s*-\s*Hôpitaux\s*de\s*Paris\s*"
        r"\(?AP-HP\)?\s*a\s*créé\s*une\s*base\s*de\s*données)"
        r".{,2000}https?:\/\/recherche\.aphp\.fr\/eds\/droit-opposition[\s\.]*"
    ),
    (
        r"(?si)l’arrêt\s*du\s*tabac\s*permet\s*de\s*diminuer\s*le\s*risque\s*"
        r"de\s*maladie\s*cardiovasculaire."
    ),
]
# Example : NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbW...
bars = r"(?i)([nbw]|_|-|=){5,}"

# Biology tables: Prone to false positive with disease names
biology = r"(\b.*[|¦].*\n)+"

# Leftside note with doctor names
doctors = r"(?mi)(^((dr)|(pr))\W.*)+"

# Mails or websites
web = [
    r"(www\.\S*)",
    r"(\S*@\S*)",
    r"\S*\.(?:fr|com|net|org)",
]

# Subsection with ICD-10 Codes
coding = r".*? \(\d+\) [a-zA-Z]\d{2,4}.*?(\n|[a-zA-Z]\d{2,4})"

pollution = dict(
    information=information,
    bars=bars,
    biology=biology,
    doctors=doctors,
    web=web,
    coding=coding,
)
