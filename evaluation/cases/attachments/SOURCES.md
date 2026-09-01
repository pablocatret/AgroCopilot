# Evaluation attachment provenance

The attachments in this directory are reproducible fixtures: cases that use
them declare their URL, licence and SHA-256 hash. They do not represent user
documentation or a real farm plot.

| File | Used by | Provenance and licence | SHA-256 |
| --- | --- | --- | --- |
| `olive_peacock_spot.jpg` | `att_001`, `att_004` | [Wikimedia Commons: *Spilocaea oleagina*](https://commons.wikimedia.org/wiki/File:Spilocaea_oleagina.jpg), dedicated to the public domain by its rights holder. | `730eef917f8727c02fec483aaf86402ca864bd935d263dfaf638ec09aa192d43` |
| `eu_organic_certificate_guidance.pdf` | `att_002`, `att_004` | [European Commission: guidance for completing the organic-production certificate](https://agriculture.ec.europa.eu/document/download/31f037c3-683b-4945-af75-a960bb1a00b6_en?filename=competent-authorities-how-model-certificate-for-organic-production_en.pdf). Public institutional document. | `5a0e35e2bd3933b5a536953cad94cdf5d892b0bbcc3479708344a5a303fb499` |
| `fao_organic_matter_guide.html` | `rt_003` | [FAO: The importance of soil organic matter](https://www.fao.org/4/a0100e/a0100e04.htm). Open FAO web publication. | `26e0c7e6495978ae3b2ae2aac32385c2d66b7b62111df322a255403190e1b2e4` |
| `rendimientos_olivar.csv` | `att_003` | Local synthetic fixture, labelled as such so it is not presented as a real observation. | `a64ab7828e0fcda915322384a35e569f62e21c0bdbb1a31e5fa263f6bd6d8ad2` |

When updating an external resource, review its licence, replace the file,
update the hash here and in the case JSON, and verify that the corpus still
responds to the actual file contents.
