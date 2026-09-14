import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


doc_prefix = "search_document: "

docs = [
    "Dubai is the most populous city in the United Arab Emirates and the capital of the Emirate of Dubai, the most populous of the country's seven emirates. As of 2024, the city has a population of around 3.79 million, more than 90% of which are expatriates.",
    "In the early 20th century, the city developed into a significant regional and international trade hub, emphasizing tourism and luxury. It has been a center for regional and international trade since that time, and its economy relies on revenues from trade, tourism, aviation, real estate, and financial services. Oil revenue helped accelerate the development of the area. However, oil production contributed less than 1 percent to the emirate's GDP in 2018. Due to financial secrecy, low taxes, and valuable real estate, the location is an appealing destination for money launderers, criminals, corrupt political figures, and sanctioned businesspeople to launder or hide money.",
    "The city has the second-most five-star hotels in the world after London and the tallest building in the world, the Burj Khalifa, which is 828 metres (2,717 ft) tall. Located in the eastern Arabian Peninsula, on the coast of the Persian Gulf, it is a major global transportation hub for passengers and cargo. In 2024, the city was the seventh most-visited in the world."
]

docs = [
    "Andrew Henry Robertson  (born 11 March 1994) is a Scottish professional footballer who plays as a left-back for Premier League club Liverpool and captains the Scotland national team. Widely regarded as one of the best left-backs in the world, he is known for his vision, athleticism and crossing ability. ",
    " Club career (1):",
    " Queen's Park (2):",
    "Focusing on studies, he was on the verge of starting university at the undergraduate level before being called up to the Queen's Park first team at the start of the 2012–13 season. Debuting for the club in the Scottish Challenge Cup during a penalty shootout victory against Berwick Rangers, he eventually made a total of forty appearances throughout the season, contributing to the club's third-place finish in the 2012–13 Scottish Third Division. He scored his first goal for the club in a 2–1 loss to East Stirlingshire on 13 November 2012.",
    " Dundee United (2):",
    "In June 2013, a player signed with Scottish Premiership club Dundee United, joining alongside Queen's Park teammate Aidan Connolly. He filled the squad position vacated by Barry Douglas, who had made the same transfer three years prior. Despite initially having their requests ignored for a transfer fee for Douglas, Queen's Park, as an amateur club, objected to losing more players to United without compensation and eventually negotiated a deal that included a percentage of future transfers, which would ultimately be advantageous.",
    "The player quickly became an integral part of manager Jackie McNamara's first team plans, making a debut for the club on the opening day of the season in a 0–0 draw with Partick Thistle. On 22 September 2013, the player scored a first goal for Dundee United in a 2–2 draw against Motherwell, running from his own half before sending a low left-foot drive into the goal from 22 yards. Shortly thereafter, a new contract with United was agreed upon, lasting until May 2016. The player was voted SPFL Young Player of the Month for September 2013 and Player of the Month for November 2013. On 12 April 2014, the player participated in the Scottish Cup Semi-final against Rangers at Ibrox, a match that United won 3–1. In April 2014, the player earned the PFA Scotland Young Player of the Year award and was also named in the PFA Scotland Team of the Year for the 2013–14 Scottish Premiership. The final appearance for United came in the 2014 Scottish Cup Final, which ended in a loss to St Johnstone.",
    " Hull City (2):",
    "In July 2014, Dundee United accepted a £2.85 million offer from English Premier League club Hull City for their player. The transfer was finalized on 29 July, with the player signing a three-year contract with Hull. Stan Ternent, Hull City's chief scout, explained, \"I'd been looking at Stuart Armstrong, but this player was the obvious choice... he had a history with Celtic and was always a determined individual, given how he had recovered from his setbacks. You could see straight away he had ability and could only improve.\"",
    "On the opening day of the season, the individual made a competitive debut in a 1–0 victory away to Queens Park Rangers, successfully executing a goal-line clearance. Quickly adjusting to life at Hull, the player secured a spot in the team and was recognized as the club's Player of the Month for August 2014. Over the course of the debut season at the KC Stadium, this person made 24 appearances, although was unable to stop the team's relegation to the Championship.",
    "Despite the departure of several other first-team players, he chose to remain at City. His inaugural goal for the club was scored on 3 November 2015 in an away match against Brentford, where he initiated the scoring in a 2–0 victory that placed Hull at the top of the Championship table on goal difference. He participated in the 2016 Championship play-off final against Sheffield Wednesday, which Hull won 1–0 to achieve promotion to the Premier League. However, the team lasted only one season in the top division before being relegated again.",
    "A proposal was made by Everton scout Steve Walsh to transfer a certain Hull defender along with his teammate Harry Maguire in a combined £20 million deal; however, Everton decided not to pursue the move.",
    " Liverpool (2):",
    " 2017–2019 (3):",
    "On 21 July 2017, a long-term contract was signed with Liverpool for an initial fee of £8 million. Debuting on 19 August, the player contributed to a 1–0 win over Crystal Palace and earned the Man of the Match award. He began the 2017–18 season as an understudy to Alberto Moreno but was given a string of opportunities when the Spaniard suffered an injury in December. The performance in the 4–3 victory against league leaders Manchester City on 14 January received acclaim from Liverpool supporters. The first goal for the club was scored on the final match day of the 2017–18 season against Brighton & Hove Albion in a 4–0 victory. Regular appearances continued during the 2018–19 season, a campaign in which Liverpool ended as Premier League runners-up. A new contract with the club was signed in January 2019, set to last until 2024. Over the course of the 2018–19 Premier League season, the player registered 11 assists. On 25 April, inclusion in the PFA Team of the Year was achieved, alongside Liverpool teammates Trent Alexander-Arnold, Sadio Mané, and Virgil van Dijk.",
    "On 1 June 2019, this player participated for the full 90 minutes of the Champions League final, contributing to Liverpool's victory over Tottenham Hotspur and securing their sixth European Cup. He became the first Scottish player to win the tournament since Darren Fletcher, who was an unused substitute in 2008, and the first to actually be on the field for a winning team since Paul Lambert in 1997.",
    " 2019–2024 (3):",
    "In the European champions' first home game of the Champions League for the 2019–20 season on 2 October, he scored his first European goal in a dramatic 4–3 victory against Red Bull Salzburg. On 2 November 2019, he netted his second Premier League goal for Liverpool, equalizing in the 87th minute against Aston Villa, with Liverpool eventually winning 2–1 in stoppage time. Throughout the 2019–20 domestic campaign, he contributed 2 goals and 12 assists for Liverpool.",
    "On 24 August 2021, a new long-term contract was signed with Liverpool.",
    "The individual received a runners-up medal in the 2021–22 UEFA Champions League after the team was defeated by Real Madrid in the final. He was also chosen as part of the Team of the Season. Liverpool came close to a historic quadruple, finishing second in both the Premier League and the 2021–22 UEFA Champions League, but secured victories in the EFL Cup and the FA Cup.",
    "During the 2022–23 season, this player became the defender with the most assists in Premier League history. By the end of the season, Liverpool narrowly missed qualifying for the UEFA Champions League.",
    "On 16 September 2023, with Virgil van Dijk suspended and Trent Alexander-Arnold injured, the player took on the role of captain in a 3–1 comeback victory against Wolverhampton Wanderers. This match marked his 200th Premier League appearance for Liverpool, during which he scored the team's second goal, his ninth goal for the club.",
]
base_model = SentenceTransformer("illuin-conteb/modernbert-large-insent")


def encode_with_offsets(texts: list[str]):
    """Encode texts and retain the character offsets needed for token alignment."""
    tokenized = base_model.tokenizer(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    offsets = tokenized.pop("offset_mapping")
    special_tokens_mask = tokenized.pop("special_tokens_mask")
    model_inputs = {name: value.to(base_model.device) for name, value in tokenized.items()}

    with torch.inference_mode():
        token_embeddings = base_model(model_inputs)["token_embeddings"].cpu()

    return tokenized["input_ids"], offsets, special_tokens_mask, token_embeddings


def main() -> None:
    tokenizer = base_model.tokenizer
    if tokenizer.sep_token is None:
        raise ValueError("The tokenizer does not define a sep_token.")

    base_model.eval()
    chunks = [doc.strip() for doc in docs]

    # Pass 1: each chunk gets its own copy of the document prefix.
    chunk_inputs = [doc_prefix + chunk for chunk in chunks]
    chunk_ids, chunk_offsets, chunk_special, chunk_embeddings = encode_with_offsets(chunk_inputs)

    # Record the content tokens from pass 1. Raw-text tokenization can change at
    # SEP boundaries when a chunk starts or ends with whitespace, so pass 2 is
    # assembled from these exact token IDs to keep a strict one-to-one mapping.
    chunk_token_records = []
    for chunk_index in range(len(chunks)):
        records = []
        for token_index, (start, end) in enumerate(chunk_offsets[chunk_index].tolist()):
            if chunk_special[chunk_index, token_index] or start == end or end <= len(doc_prefix):
                continue
            records.append(
                (
                    token_index,
                    (max(start, len(doc_prefix)) - len(doc_prefix), end - len(doc_prefix)),
                    int(chunk_ids[chunk_index, token_index]),
                )
            )
        chunk_token_records.append(records)

    # Pass 2: [prefix tokens] chunk 1 [SEP] chunk 2 ... in one model call.
    prefix_ids = tokenizer(doc_prefix, add_special_tokens=False)["input_ids"]
    document_content_ids = list(prefix_ids)
    document_positions = []
    for chunk_index, records in enumerate(chunk_token_records):
        if chunk_index:
            document_content_ids.append(tokenizer.sep_token_id)
        positions = []
        for _, _, token_id in records:
            positions.append(len(document_content_ids))
            document_content_ids.append(token_id)
        document_positions.append(positions)

    wrapped_prefix_ids = tokenizer(doc_prefix, add_special_tokens=True)["input_ids"]
    prefix_start = next(
        start
        for start in range(len(wrapped_prefix_ids) - len(prefix_ids) + 1)
        if wrapped_prefix_ids[start : start + len(prefix_ids)] == prefix_ids
    )
    leading_special_ids = wrapped_prefix_ids[:prefix_start]
    trailing_special_ids = wrapped_prefix_ids[prefix_start + len(prefix_ids) :]
    document_ids = leading_special_ids + document_content_ids + trailing_special_ids
    content_start = len(leading_special_ids)
    model_inputs = {
        "input_ids": torch.tensor([document_ids], device=base_model.device),
        "attention_mask": torch.ones((1, len(document_ids)), dtype=torch.long, device=base_model.device),
    }
    with torch.inference_mode():
        document_embeddings = base_model(model_inputs)["token_embeddings"].cpu()[0]

    print("chunk\tchars\ttext\ttoken\tcosine_similarity")
    compared_tokens = 0
    for chunk_index, chunk in enumerate(chunks):
        records = chunk_token_records[chunk_index]
        for record_index, (token_index, local_offset, token_id) in enumerate(records):
            document_index = content_start + document_positions[chunk_index][record_index]
            second_embedding = document_embeddings[document_index]
            first_embedding = chunk_embeddings[chunk_index, token_index]
            similarity = F.cosine_similarity(
                first_embedding.float(), second_embedding.float(), dim=0
            ).item()
            token = tokenizer.convert_ids_to_tokens(token_id)
            text = chunk[slice(*local_offset)]
            print(
                f"{chunk_index + 1}\t{local_offset[0]}:{local_offset[1]}\t"
                f"{text!r}\t{token!r}\t{similarity:.6f}"
            )
            compared_tokens += 1

    expected_tokens = sum(len(records) for records in chunk_token_records)
    if compared_tokens != expected_tokens:
        raise RuntimeError(f"Aligned {compared_tokens} of {expected_tokens} content tokens.")


if __name__ == "__main__":
    main()
