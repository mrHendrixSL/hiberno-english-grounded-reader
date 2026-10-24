# hiberno-english-grounded-reader

Extraction pipeline, schema definitions, and evaluation code for a
hybrid (rule-based + LLM) structured-data extraction system applied to
*A Dictionary of Hiberno-English* (Dolan, 2020, 3rd edn, Gill Books).

This repository is the public, reconstructed companion to:

- the **journal article** describing the method (citation / DOI to
  follow once published), and
- the **dataset** itself, deposited on Zenodo under CC BY-NC-SA 4.0
  (DOI to follow once minted) — the structured output this code
  produces, not reproduced here.

It does not contain the copyrighted source text. The source edition is:

> Dolan, T. P. (2020). *A Dictionary of Hiberno-English: The Irish Use
> of English*. 3rd edn, with a foreword by Blindboy Boatclub. Dublin:
> Gill Books. ISBN 978-0-7171-9020-1

## Status

This repository is being assembled commit-by-commit, mirroring the
original development sequence (ingestion → apparatus extraction →
segmentation → schema → rule-based reader → LLM readers → evaluation →
hybrid production). It currently contains project scaffolding only; the
pipeline code lands in subsequent commits.

## Method, in brief

A deterministic rule-based parser and a panel of LLM readers each
extract the same schema independently. A frozen, pre-registered
per-field routing decision — selected on a 390-entry development sample
(E390) before any held-out scoring happened — picks whichever system
scored best for each field group, producing the **HYBRID** dataset.
Accuracy is reported against two reference samples of different
strength (E390, machine-led; H390, human-only and held out). Full
methodology and numbers are in the dataset's own `ACCURACY.md`, and in
the paper.

## Licence

Code in this repository is MIT-licensed — see `LICENSE`. The dataset it
produces has its own licence (CC BY-NC-SA 4.0); the source text is used
under permission and is not redistributed, in whole or in
representative part, under any licence granted by this repository.

## Citation

See `CITATION.cff` for how to cite this code. To cite the dataset, use
the citation given in the Zenodo deposit's own README.
