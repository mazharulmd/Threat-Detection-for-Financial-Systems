# Reference verification log (2026-11-01)

The direct bibliographic APIs (OpenAlex, Crossref, arXiv, Semantic Scholar) were blocked from the
build sandbox, so each entry was checked through the Consensus index (Semantic Scholar / OpenAlex
records) and web search of publisher, DBLP and repository pages. Rule followed: **only fields that
were confirmed are included**; where volume/pages could not be confirmed they are omitted and
author lists are shortened to "et al.".

| Key | Status | Notes |
|---|---|---|
| ibm2024breach | verified | Financial sector average US$6.08M in 2024 (2nd highest of 17 industries) — several independent reports of the IBM study |
| asmar2024banks | verified (index) | Heliyon 2024; co-authors not confirmed → "et al." |
| islam2022ddosbank | verified (index) | Sustainability 2022 |
| bhowmick2026cbs | verified (index) | Scientific Reports 2026 |
| mulea2026banking | verified (index) | ICCWS 2026 |
| sharafaldin2018cicids, moustafa2015unsw | verified | standard dataset papers |
| yang2022mthids | venue/year verified | vol./pages not confirmed → omitted. Source of the CICIDS2017 subset used |
| engelen2021troubleshooting | verified (index) | IEEE SPW 2021 |
| catillo2022transferability | verified (index) | Software Quality Journal 2022 |
| arp2022dos | verified | USENIX Security 2022, pp. 3971–3988 |
| halbouni2022cnnlstm | verified | IEEE Access vol. 10, 2022 (authors from IIUM repository); pages not confirmed |
| altunay2023cnnlstm | verified (index) | Eng. Sci. Technol. Int. J. 2023 |
| sajid2024hybrid | verified | J. Cloud Computing 13(1), 2024, DOI 10.1186/s13677-024-00685-x |
| alashjaee2025attn | verified (index) | Scientific Reports 2025 |
| wu2022rtids | verified | IEEE Access 10:64375–64387, DOI 10.1109/ACCESS.2022.3182333 |
| manocchio2024flowtransformer | verified | ESWA 241:122564 (author list differs between preprint and journal → "et al.") |
| ullah2023idsint, long2024cloud, akuthota2025iot, yang2025bert | verified (index) | venue/year only |
| gorishniy2021revisiting, grinsztajn2022tree | verified | NeurIPS 2021; NeurIPS 2022 Datasets & Benchmarks |
| neupane2022xids, barnard2022xai | verified (index) | IEEE Access 2022; IEEE Networking Letters 2022 |
| lefevre2006mondrian | verified | ICDE 2006, p. 25 |
| bun2016zcdp | venue/year verified | TCC 2016-B; pages not confirmed → omitted |
| ruzafa2023fl, vyas2024ppfl, gutti2025fl | verified (index) | IEEE TII 2023; IEEE Access 2024; IEEE Access 2025 |
| classics (Hochreiter 1997, Vaswani 2017, Chen 2016, Chawla 2002, Breiman 2001, Liu 2008, Sweeney 2002, Dwork & Roth 2014, Abadi 2016, McMahan 2017) | well-established | standard bibliographic data |

## Removed from the original draft

| Original citation | Reason |
|---|---|
| K. Yang et al., "BERT-IDS ...", IEEE TIFS 2022 | not found in any index — likely non-existent; replaced by Yang et al., EURASIP JIS 2025 (verified) |
| P. Kaur et al., "Compliance-aware security monitoring for financial systems", IEEE CloudCom 2023 | not found — removed |
| M. Hashem et al., "Insider threat detection using ML: A comprehensive survey", Comput. Secur. 2021 | could not be verified — removed |
| Y. Kim and H. Kim, "CNN-LSTM based intrusion detection for uncommon attacks", Electronics 2020 | could not be verified — replaced by Halbouni et al. 2022 (verified) |
| M. Ficco and F. Palmieri, Android malware families, 2018 | off-topic for this paper — removed |

**Before submission**, please look up the remaining "venue/year only" entries on IEEE Xplore / the
publisher site and add volume, pages and DOI.
