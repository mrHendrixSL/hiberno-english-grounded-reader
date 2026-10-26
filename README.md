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

This repository is being assembled commit-by-commit. `notebooks/pipeline.ipynb`
runs the pipeline stage by stage — ingestion, apparatus extraction,
segmentation, structural sampling (E390/H390), deterministic resolution,
the rule-based reader, the LLM readers, structural retrieval, and
evaluation scoring are in place; hybrid production assembly is still to
come. See the notebook itself for the current stage list.

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

## Contributors

- Rasika Edirisinghe (University College Cork)
- James O'Sullivan (University College Cork)
- Rosane Minghim (University College Cork)
- Órla Murphy (University College Cork)

## Citation

If you use this code, please cite it:

> Edirisinghe, R., O'Sullivan, J., Minghim, R., & Murphy, Ó. (2026).
> *hiberno-english-grounded-reader* [Computer software].
> https://github.com/mrHendrixSL/hiberno-english-grounded-reader

See `CITATION.cff` for the machine-readable version. To cite the
dataset itself, use the citation given in the Zenodo deposit's own
README instead.
