#!/usr/bin/env python3
"""Compare independent and contextual token embeddings for one document.

The two paths use the same base model.  The independent path encodes every
chunk on its own; the contextual path encodes ``sep_token.join(chunks)`` once.
For chunks after the first, the independently encoded leading special token is
not part of the joined document and is therefore excluded from the comparison.

Run without arguments; edit ``MODEL_NAME`` or ``DEFAULT_CHUNKS`` below to
change the model and the batch of hard-coded document chunks.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

import torch
from sentence_transformers import SentenceTransformer
from torch.nn import functional

# Let the script run from either repository root or the playground directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTEB_ROOT = REPOSITORY_ROOT / "conteb"
CONTEXTUAL_EMBEDDINGS_ROOT = REPOSITORY_ROOT / "contextual-embeddings"
if str(CONTEB_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEB_ROOT))
if str(CONTEXTUAL_EMBEDDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTEXTUAL_EMBEDDINGS_ROOT))

from cde_benchmark.embedders.contextual_embedder import ContextualEmbedder  # noqa: E402, I001
from cde_benchmark.embedders.sentence_transformer_embedder import (  # noqa: E402
    SentenceTransformerEmbedder,
)


LIGHTHOUSE_EXAMPLE = [
    "The old lighthouse keeper trimmed his lamp, its beam cutting a lonely path through the fog.",
    "He remembered nights of violent storms, when the ocean seemed to swallow the sky whole.",
    "Still, he found comfort in his duty, a silent guardian against the treacherous sea."
]

MODEL_NAME = "lightonai/modernbert-embed-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


INSURANCE_EXAMPLE = [
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Austria\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000)³     | 8,979           | 2.00%          |\n| (Re)insurance GWP (in million)⁴ | 20,815.873      | 1.5%           |\n| Number of (re)insurance undertakings⁵ | 33              | 1.9%           |\n| Number of registered insurance intermediaries | 17,999          | 2.1%           |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Belgium\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000)⁹     | 11,618          | 2.6%           |\n| (Re)insurance GWP (in million)¹⁰ | 44,172.414       | 3.2%           |\n| Number of (re)insurance undertakings¹¹ | 57              | 3.3%           |\n| Number of registered insurance intermediaries | 8,911          | 1.1%           |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Bulgaria\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000) | 6,839 | 1.5% |\n| (Re)insurance GWP (in million) | 2,730.334 | 0.2% |\n| Number of (re)insurance undertakings | 30 | 1.7% |\n| Number of registered insurance intermediaries | 6,839 | 0.8% |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Croatia\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000) | 4,035.65 | 0.8% |\n| (Re)insurance GWP (in million) | 1,353 | 0.1% |\n| Number of (re)insurance undertakings | 23 | 1.7% |\n| Number of registered insurance intermediaries | 3,985 | 0.5% |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Cyprus\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000) | 904.705 | 0.2% |\n| (Re)insurance GWP (in million) | 1,217.479 | 0.09% |\n| Number of (re)insurance undertakings | 29 | 1.7% |\n| Number of registered insurance intermediaries | 2,322 | 0.3% |'}",
    "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Czechia\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000) | 10,517 | 2.3% |\n| (Re)insurance GWP (in million) | 8,386.338 | 0.6% |\n| Number of (re)insurance undertakings | 22 | 1.3% |\n| Number of registered insurance intermediaries | 32,448 | 3.7% |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Denmark\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000)      | 5,873            | 1.3%             |\n| (Re)insurance GWP (in million) | 40,744.612     | 3.0%             |\n| Number of (re)insurance undertakings | 72           | 4.2%             |\n| Number of registered insurance intermediaries | 2,571 | 0.3%             |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Estonia\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                    | Share total EEA |\n|---------------------------|-----------------|\n| Population (in 1000)            | 1,332            | 0.3%             |\n| (Re)insurance GWP (in million)  | 956,810          | 0.07%            |\n| Number of (re)insurance undertakings | 9           | 0.5%             |\n| Number of registered insurance intermediaries | 394   | 0.05%            |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Finland\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                       | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 5,548            | 1.2%              |\n| (Re)insurance GWP (in million)| 8,449.683        | 0.62%             |\n| Number of (re)insurance undertakings | 43            | 2.5%              |\n| Number of registered insurance intermediaries | 3,812 | 0.4%              |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# France\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                       | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 67,872           | 15%               |\n| (Re)insurance GWP (in million)| 330,940.978      | 24.3%             |\n| Number of (re)insurance undertakings | 151        | 8.7%              |\n| Number of registered insurance intermediaries | 62,284 | 7.2%             |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Germany\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                       | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 83,237           | 18.4%             |\n| (Re)insurance GWP (in million)| 335,298.764      | 24.7%             |\n| Number of (re)insurance undertakings | 256          | 14.8%              |\n| Number of registered insurance intermediaries | 190,708       | 22%              |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Greece\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                       | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 10,460           | 2.3%             |\n| (Re)insurance GWP (in million)| 4,687.315        | 0.4%             |\n| Number of (re)insurance undertakings | 34               | 2.0%             |\n| Number of registered insurance intermediaries | 17,164      | 2%               |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Hungary\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                      | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 9,689            | 2.1%              |\n| (Re)insurance GWP (in million)| 3,562.353        | 0.3%              |\n| Number of (re)insurance undertakings | 22          | 1.3%              |\n| Number of registered insurance intermediaries | 45,376      | 5.2%              |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Iceland\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                           | Share total EEA |\n|----------------------------------|------------------|\n| Population (in 1000)             | 376.248          | 0.08%            |\n| (Re)insurance GWP (in million)   | 572.893          | 0.04%            |\n| Number of (re)insurance undertakings | 8         | 0.5%             |\n| Number of registered insurance intermediaries | 54      | 0.01%            |'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Ireland\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                      | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 5,060            | 1.18% |\n| (Re)insurance GWP (in million) | 102,420.675       | 7.5% |\n| Number of (re)insurance undertakings | 163          | 9.4% |\n| Number of registered insurance intermediaries | 2,583   | 0.3% |\n'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Italy\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                      | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 59,030            | 13% |\n| (Re)insurance GWP (in million) | 135,019.570      | 9.9% |\n| Number of (re)insurance undertakings | 87         | 5.0% |\n| Number of registered insurance intermediaries | 235,404  | 27.2% |\n'}",
    # "COUNTRY-BY-COUNTRY ANALYSIS – Annex VII to the 2nd Report on the application of the IDD\n\n# Latvia\n### Information on the insurance intermediaries' market structure and patterns of cross-border activity\n",
    # "{'General data': '| Amounts                      | Share total EEA |\n|-------------------------------|------------------|\n| Population (in 1000)          | 1,876            | 0.4% |\n| (Re)insurance GWP (in million) | 623,982          | 0.05% |\n| Number of (re)insurance undertakings | 6         | 0.3% |\n| Number of registered insurance intermediaries | 1,397  | 0.2% |\n'}",
]

FOX_EXAMPLE = [
    "A curious fox cub, all rust and wonder, ventured out from its den for the first time.",
    "Each rustle of leaves, every chirping bird, was a new symphony to its tiny ears.",
    "Under the watchful eye of its mother, it began to learn the secrets of the whispering forest."
]

COVID_EXAMPLE = [
    "First Complete Genome Sequence of a French Bovine coronavirus Strain\n\nhttps://www.ncbi.nlm.nih.gov/pmc/articles/PMC5477389/\n\nSHA: eef0ecf5b8e7b179dadaef967e65f2ab68f021e1\n\nAuthors: Kin, Nathalie; Guerard, Pauline; Diancourt, Laure; Caro, Valérie; Vabret, Astrid; Ar Gouilh, Meriadeg\nDate: 2017-05-25\nDOI: 10.1128/genomea.00319-17\nLicense: cc-by\n\nAbstract: We sequenced the first Bovine coronavirus (BCoV) complete genome sequence from France. This BCoV was directly sequenced from a fecal sample collected from a calf in Normandy in 2014.",
    "Text: B ovine coronavirus (BCoV) belongs to the Nidovirales order, the Coronaviridae family, the Coronavirinae subfamily, and the Betacoronavirus (https://talk.ictvonline.org/ ICTV/proposals/2008.085-122V.v4.Coronaviridae.pdf). Its genome is a single-stranded, linear, and nonsegmented RNA of around 31 kb. BCoV is responsible for respiratory and enteric diseases in cattle, particularly during winter (1, 2) . To date, the 19 complete BCoV genome sequences available in GenBank databases (consulted on 17 January 2017) originated from the United States or Asia. Here, we report the first complete genome sequence of a BCoV detected in France.",
    "The BCoV/FRA-EPI/CAEN/2014/13 strain was obtained from a fecal sample collected from a 1-week-old calf in Normandy in 2014. The presence of BCoV in the fecal sample was assessed using an in-house reverse transcription-PCR (RT-PCR) targeting the M gene (3). A cDNA library was synthesized using SuperScript III (Invitrogen, Carlsbad, CA, USA) and hexamers. The complete genome sequencing of overlapping PCR products was carried out in both directions, using original primers and Sanger's dideoxy sequencing. Sequencing reactions were performed as previously described (3). Sequences were assembled and annotated using the Geneious software (version 5.1.6). We obtained a sequence counting 30,847 nucleotides. The orf1ab, HE, S, ns5, E, M, and N genes of the obtained BCoV were submitted to a Blastn analysis. According to these analyses, the orf1ab (20kb nucleotides, located at the 5= side of the genome) gene is closely related to the Dromedary camel coronavirus (DcCoV) HKU23-23-362F strain from",
    "the United Arab Emirates (accession no. KF906251), with a nucleotide identity of 99.19%. Conversely, the NS2, HE, S, ns5, and M genes are closely related to the BCoV Bubalus/Italy/179/07-11 strain (accession no. EU019216), with nucleotide identities of 99.88%, 99.45%, 99.02%, 98.79%, and 99.28%, respectively. The E gene is closely related to the Chinese Bovine coronavirus strain BCV-AKS-01 (accession no. KU886219), with a nucleotide identity of 100%. Finally, the highest Blastn score for the N gene was found with the American enteric BCoV-ENT (accession no. AF391541), associated with a nucleotide identity of 100%.",
    "Multiple-sequence alignment, including 20 BCoVs and 10 clade A betacoronaviruses closely related to BCoV from North America, two DcCoVs from the United Arab Emirates, and two Human coronavirus OC43 (HCoV-OC43) strains from France, was performed using the Muscle algorithm implemented in MEGA7 (4, 5) . The phylogenetic analysis on the orf1ab confirms that BCoV/FRA-EPI/CAEN/2014/13 is closely related to the Dromedary camel coronavirus (DcCoV) HKU23-23-362F. The orf1ab gene of these two viruses together clustered separately from that of BCoV and BCoV-like viruses from North America and Asia. This finding also confirms the results from our previous analysis on partial genomes in which nsp12, S, and N genes of American and Asian BCoVs group together in a cluster tentatively named C 1 . The nsp12 and N coding regions of BCoVs from France and DcCoVs from the United Arab Emirates clustered together in C 2 . The DcCoV S gene individualized from both HCoV-OC43 and BCoV S genes. Potential",
    "recombination events could be at the origin of DcCoV.",
    "Accession number(s). The complete genome sequence sequence of the BCoV/FRA-EPI/CAEN/2014/13 isolate has been deposited in GenBank under the accession number KX982264."
]

MLDR_EXAMPLE =  [
    """Lindalva Justo de Oliveira (20 October 1953 - 9 April 1993) was a Brazilian Roman Catholic professed religious and a professed member of the Vincentian Sisters. Oliveira worked at, Dom Pedro II Home, a retirement home, where she was killed in 1993 after a man obsessed with her stabbed her 44 times when she refused his unwanted advances.

Oliveira's beatification received the approval of Pope Benedict XVI who determined that she was killed "in defensum castitatis" - the defense of her vow for being chaste. She was beatified on 2 December 2007 in which Cardinal José Saraiva Martins presided over on the behalf of the pope.

Life
Lindalva Justo de Oliveira was born on 20 October 1953 in Brazil as the sixth of thirteen children to the farmer João Justo da Fé - a widower with three children - and Maria Lúcia da Fé (b. 1923); two brothers were Antonio and Djamla. As of 2014 her mother was still alive at age 91. Antonio was an alcoholic and went sober not long after she penned a letter to him.""",
    """Oliveira received her baptism on 7 January 1954 in the parish of Saint John the Baptist in the Chapel of Olho D'Água from Monsignor Júlio Alves Bezerra. Around 1961 her parents took their children elsewhere so as to provide for their educational needs.

She received her First Communion on 15 December 1965 and as of 1971 began to help raise her three nephews. Her parents tried to convince her to wed at this time but she deflected the conversation and instead said that he had three sons: her brother's three children that she helped care for. Oliveira lived with her brother Djamla in Natal for a time and received an administrative assistant's diploma in 1979. From 1978 until a decade later she worked in retail sales and also as a cashier at a petrol station. This was to provide for her ailing father and her mother and after the death of her father was for her mother's financial situation; leftover wages were for her personal use and she lived in Natal during this time.""",
    """As her father was on his deathbed in 1982 she aided him in his last months. He later summoned his children to him while asking a priest for the Anointing of the Sick; he beseeched his children to persevere in the faith and devote themselves as best as possible to God. He died hours later in 1982 due to abdominal cancer. Not long after in 1982 she began a technical course in nursing. In 1986 she attended a vocational movement of the Vincentian Sisters and requested joining them at the end of 1987; the Archbishop of Natal Nivaldo Monte granted her the sacrament of Confirmation on 28 November 1987.

On 28 December 1987 she received a letter from the mother provincial accepting her entrance into the congregation. Her time as a postulant commenced on 11 February 1988 in Recife and she requested to commence her novitiate on 3 June 1989. She and five other hopefuls commenced their novitiate on 16 July 1989.""",
    """She began to work at the shelter titled Don Pedro II Home in Bahia on 29 January 1991 to aid older people and the poor; she went on a retreat that same month while the forum was dedicated to the charism of Saint Vincent de Paul. Oliveira even took a driving test so she could take some of the people from the shelter for rides. In 1993 a man named Augusto da Silva Peixoto (b. 1947) was admitted into the shelter despite the fact that he was not meant to be there at all; he became obsessed with Oliveira and began to harass her despite her best efforts to keep her distance from him while treating him like she did the others. Those around her convinced her to report it and on 30 March 1993 the official of the shelter - Margarita Maria Siva de Azevedo - rebuked him; Augusto responded on 5 April in purchasing a machete.""",
    """On 9 April 1993 she participated in the Way of the Cross at 4:30am and returned to the shelter at 7:00am to serve breakfast at the Dom Pedro II Home. Augusto approached her as she served coffee and tapped her on the shoulder before thrusting a knife into her above the collar-bone as she turned around. She sank to the ground and cried out several times: "God protect me" while her attacker shouted: "I should have done this sooner!" A man attempted to intervene but Augusto warned those who approached would be killed. Augusto wiped the knife of blood on his clothes and threw it onto the floor before exclaiming to the horrified witnesses: "She did not want me!" He then said to the doctor that was summoned: "You can call the police, I will not run away; I did what had to be done". Augusto's reason for killing her was due to Oliveira refusing to give up the religious life to be with him as a lover. The killer sat on a bench outside the shelter and awaited the police; he was admitted to a""",
    """mental hospital following his conviction. Coroners identified a total of 44 perforations in Oliveira.""",
    """Augusto was still alive as of 2007. He was in a mental hospital until 2005.

Her funeral was celebrated on 10 April 1993 and the Dominican Cardinal Lucas Moreira Neves presided over the funeral. As of 6 April 2014 her remains are in the Capela das Relíquias da Beata Lindalva.

Beatification

The beatification process commenced in Brazil on 19 October 1999 - under Pope John Paul II - after the Congregation for the Causes of Saints declared "nihil obstat" ('nothing against') to the cause while also acknowledging her as a Servant of God. The diocesan process opened on 17 January 2000 and concluded its business not long after on 3 March 2001; the C.C.S. validated this process in Rome on 22 June 2001.""",
    """The Positio was sent to the C.C.S. in 2002 at which point it was retained until theologians met and approved the cause in a meeting on 26 September 2006; the C.C.S. also granted their approval to the merits of the cause on 21 November 2006. On 16 December 2006 her beatification received the papal approval of Pope Benedict XVI who confirmed she was killed "in defensum casitatis".

The beatification celebration was held in Brazil on 2 December 2007 and Cardinal José Saraiva Martins presided as the delegate the pontiff appointed in his stead.

References

External links
 Hagiography Circle
 Saints SQPN

1953 births
1993 deaths
1993 crimes
20th-century venerated Christians
20th-century Roman Catholic martyrs
20th-century Brazilian people
Beatifications by Pope Benedict XVI
Brazilian beatified people
Brazilian murder victims
Brazilian women
Deaths by stabbing in Brazil
People from Rio Grande do Norte
People murdered in Brazil
People executed by stabbing
Venerated Catholics""",
    """The success of NISS writing workshops is partially evident in the success of workshop graduates. Many workshop graduates are serving on the editorial boards of major statistical and biostatistical journals, including Annals of Statistics, Journal of the American Statistical Association, Journal of the Royal Statistical Society, Technometrics, Journal of Computational and Graphical Statistics, Computational Statistics & Data Analysis, Journal of Multivariate Analysis, Bernoulli, Statistica Sinica, Electronic Journal of Statistics, Journal of Statistical Planning and Inference, Statistics and Its Interface, Journal of Statistical Computation and Simulation, Statistics in Medicine and Statistical Communications in Infectious Diseases.

References

External links 

Mathematics organizations
Statistical organizations
Scientific organizations established in 1990"""
]
EXAMPLES = [
    ("lighthouse", LIGHTHOUSE_EXAMPLE),
    ("insurance", INSURANCE_EXAMPLE),
    ("fox", FOX_EXAMPLE),
    ("covid", COVID_EXAMPLE),
    ("mldr", MLDR_EXAMPLE),
]

_MODEL_CACHE = None


def model_token_embeddings(model: SentenceTransformer, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded token embeddings and their attention mask for ``texts``."""
    features = model.tokenize(texts)
    tensor_features = {
        key: value.to(model.device)
        for key, value in features.items()
        if isinstance(value, torch.Tensor)
    }
    with torch.inference_mode():
        embeddings = model(tensor_features)["token_embeddings"]
    return embeddings, tensor_features["attention_mask"].bool()


def valid_rows(
    token_ids: torch.Tensor,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Iterable[tuple[int, torch.Tensor]]:
    """Yield unpadded ``(token_id, embedding)`` rows in sequence order."""
    for token_id, embedding, is_valid in zip(token_ids, embeddings, attention_mask):
        if is_valid:
            yield token_id.item(), embedding


def colorize_token(token: str, score: float, low: float, high: float, enabled: bool) -> str:
    """Color a token from red at the chunk minimum to green at its maximum."""
    if not enabled:
        return token
    ratio = 0.5 if high == low else (score - low) / (high - low)
    ratio = max(0.0, min(1.0, ratio))
    red = round(255 * (1.0 - ratio))
    green = round(180 * ratio)
    return f"\033[38;2;{red};{green};0m{token}\033[0m"


def readable_text(text: str) -> str:
    """Remove undecodable and terminal-control characters from decoded text."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return re.sub(r"\s+", " ", text).replace("�", "")


def render_chunk(
    tokenizer,
    chunk_scores: list[tuple[int, float]],
    low: float,
    high: float,
    use_color: bool,
) -> str:
    """Render a chunk as continuous text, coloring each visible token."""
    pieces = []
    decoded_prefix = ""
    token_ids = []
    for token_id, similarity in chunk_scores:
        if tokenizer.convert_ids_to_tokens(token_id) in tokenizer.all_special_tokens:
            continue
        token_ids.append(token_id)
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        # Incremental decoding preserves the tokenizer's own spacing and
        # subword rules (for example, COUNTRY is not rendered as C OUN TRY).
        text = decoded[len(decoded_prefix) :]
        decoded_prefix = decoded
        text = readable_text(text)
        if text:
            pieces.append(colorize_token(text, similarity, low, high, use_color))
    return "".join(pieces)


def run_example(
    example_name: str,
    example_chunks: list[str],
    use_color: bool | None = None,
) -> None:
    global _MODEL_CACHE
    # A trailing newline is ignored when a chunk is encoded independently but
    # becomes an actual token when followed by the literal separator string in
    # the joined document.  Normalize boundaries so both paths tokenize the
    # same text.
    chunks = [chunk.strip() for chunk in example_chunks]
    if not all(chunk.strip() for chunk in chunks):
        raise ValueError("Chunks must be non-empty strings.")

    if _MODEL_CACHE is None:
        base_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
        sentence_embedder = SentenceTransformerEmbedder(
            model=base_model, show_progress_bar=False, device=DEVICE
        )
        contextual_embedder = ContextualEmbedder(
            model=base_model, show_progress_bar=False, device=DEVICE
        )
        _MODEL_CACHE = (base_model, sentence_embedder, contextual_embedder)
    base_model, sentence_embedder, contextual_embedder = _MODEL_CACHE

    # The wrapper APIs return pooled chunk vectors.  Their enclosed base models
    # expose the required per-token vectors, while still exercising the two
    # wrapper instances specified above.
    independent_embeddings, independent_mask = model_token_embeddings(
        sentence_embedder.model, chunks
    )
    independent_ids = sentence_embedder.model.tokenize(chunks)["input_ids"]

    sep_token = contextual_embedder.base_model.tokenizer.sep_token
    if sep_token is None:
        raise ValueError("The selected tokenizer has no sep_token and cannot join chunks this way.")
    joined_document = sep_token.join(chunks)
    contextual_embeddings, contextual_mask = model_token_embeddings(
        contextual_embedder.base_model, [joined_document]
    )
    contextual_ids = contextual_embedder.base_model.tokenize([joined_document])["input_ids"][0]

    # A separately encoded BERT-style chunk starts with CLS.  The joined
    # sequence contains that token only once, before the first chunk.
    independent_rows: list[tuple[int, int, torch.Tensor]] = []
    for chunk_index in range(len(chunks)):
        rows = list(valid_rows(
            independent_ids[chunk_index],
            independent_embeddings[chunk_index],
            independent_mask[chunk_index],
        ))
        # Keep the first chunk's leading special token and remove the repeated
        # leading special token from later independently encoded chunks.  The
        # latter is already present once at the start of the joined sequence.
        if chunk_index and rows and rows[0][0] in {
            token_id
            for token_id in (
                tokenizer_id
                for tokenizer_id in (
                    sentence_embedder.model.tokenizer.cls_token_id,
                    sentence_embedder.model.tokenizer.bos_token_id,
                )
                if tokenizer_id is not None
            )
        }:
            rows = rows[1:]
        independent_rows.extend((chunk_index, token_id, embedding) for token_id, embedding in rows)

    contextual_rows = list(
        valid_rows(contextual_ids, contextual_embeddings[0], contextual_mask[0])
    )
    tokenizer = base_model.tokenizer
    independent_token_ids = [token_id for _, token_id, _ in independent_rows]
    contextual_token_ids = [token_id for token_id, _ in contextual_rows]
    if independent_token_ids != contextual_token_ids:
        raise RuntimeError(
            "Token sequences do not align. This experiment assumes that separately "
            "encoded chunks have one leading special token and use the same sep_token."
        )

    print(f"example: {example_name}")
    print(f"model: {MODEL_NAME}")
    print(f"device: {DEVICE}")
    print(f"separator: {sep_token!r}")
    scores = []
    for (chunk_index, token_id, independent), (_, contextual) in zip(
        independent_rows, contextual_rows
    ):
        similarity = functional.cosine_similarity(independent.float(), contextual.float(), dim=0).item()
        scores.append((chunk_index, token_id, similarity))

    ranges = {}
    for chunk_index, _, similarity in scores:
        low, high = ranges.get(chunk_index, (similarity, similarity))
        ranges[chunk_index] = (min(low, similarity), max(high, similarity))

    # Notebook output is not a TTY, so callers can explicitly enable ANSI colors.
    if use_color is None:
        use_color = os.environ.get("NO_COLOR") is None and sys.stdout.isatty()
    print("\ncontinuous paragraphs by chunk:")
    for chunk_index, _ in enumerate(chunks):
        chunk_scores = [
            (token_id, similarity)
            for score_chunk, token_id, similarity in scores
            if score_chunk == chunk_index
        ]
        low, high = ranges[chunk_index]
        paragraph = render_chunk(tokenizer, chunk_scores, low, high, use_color)
        print(f"chunk {chunk_index}: {paragraph}")


def main() -> None:
    for example_name, example_chunks in EXAMPLES:
        run_example(example_name, example_chunks)
        print()


if __name__ == "__main__":
    main()
