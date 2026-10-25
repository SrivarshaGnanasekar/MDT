# Reference verification follow-up

Checked 24 October 2026 against the supplied `references.txt`. This is a bibliography and project-fit check; the manuscript files were not opened.

## Remaining checks in the supplied list

- **#42 is verified.** Correct DOI: `10.1002/uog.10082`. Full citation: Sarris I, Ioannou C, Chamberlain P, Ohuma E, Roseman F, Hoch L, Altman DG, Papageorghiou AT, for the International Fetal and Newborn Growth Consortium for the 21st Century. Intra- and interobserver variability in fetal ultrasound measurements. *Ultrasound in Obstetrics & Gynecology*. 2012;39(3):266–273. The article reports interobserver AC limits of agreement of ±8.8% (or ±1.35 Z-score), so cite it for measurement variability rather than a universal fixed millimetre tolerance. [Wiley full text](https://obgyn.onlinelibrary.wiley.com/doi/full/10.1002/uog.10082) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/22535628/).
- **#11 is verified as a published online-first article.** The publisher lists all authors as Jiao Wang, Xianmin Wang, Yanmin Yu, Wei Xiang, Ying Li, XiaoFeng Zhang, Qian Sun, Yu He, Jianqun Li, and Hongyu Yuan. DOI `10.1038/s41598-026-65628-9`; published 8 October 2026. On the publisher page checked on 24 October, no volume, issue, or article number had been assigned. Cite as *Scientific Reports* (2026), online ahead of print, DOI above; update the volume/article number when the Version of Record supplies it. [Publisher page](https://www.nature.com/articles/s41598-026-65628-9).
- **URL-only references #19, #26–30, #36, and #41:** the linked arXiv or publisher proceedings pages opened and their titles/authors match the entries. Keep the supplied official links. **#5 (NICE NG3):** the supplied address is the official guideline URL, but NICE returned HTTP 403 to the checker, so page loading could not be independently confirmed here. Bibliographic title and guideline number are consistent with NICE NG3. [NICE NG3](https://www.nice.org.uk/guidance/ng3).

## Resolved citation gaps and fit notes

### 1. Choi median AC equation

The “Choi et al.” citation is real and directly matches the equation in `fetal/ac_predict.py`. Correct citation: Choi HY, Kim JH, Park JY, Jung EY, No JH, Oh KJ, Hong JS. Simple mathematical formulae for estimation of median values of fetal biometry at each gestational age. *Obstetrics & Gynecology Science*. 2016;59(2):91–96. DOI: [10.5468/ogs.2016.59.2.91](https://doi.org/10.5468/ogs.2016.59.2.91). It reports the fitted median AC equation `AC (cm) = 1.013 × GA (weeks) − 4.634`.

There is a use-range caveat: the equation was derived from reference values after 20 weeks and compared with observed data from a Korean cohort at 20–40 weeks. The project predictor comment says `GA >= 14 weeks`, which this source does not support. Treat the equation as a rough Korean median approximation, not a universal clinical reference or centile curve. If used from 14–19 weeks or for a different population, select and validate a suitable gestational-age reference instead.

Other actual Korean population charts, if needed for a more appropriate population-based reference, are Jung SI, Lee YH, Moon MH, et al. *Prenatal Diagnosis*. 2007;27(6):545–551. DOI: [10.1002/pd.1729](https://doi.org/10.1002/pd.1729); and Kwon JY, Park IY, Wie JH, Choe S, Kim CJ, Shin JC. *Prenatal Diagnosis*. 2014;34(10):927–934. DOI: [10.1002/pd.4394](https://doi.org/10.1002/pd.4394).

### 2. INTERGROWTH-21st estimated fetal weight

Stirnemann J, Villar J, Salomon LJ, et al. International estimated fetal weight standards of the INTERGROWTH-21st Project. *Ultrasound in Obstetrics & Gynecology*. 2017;49(4):478–486. DOI: [10.1002/uog.17347](https://doi.org/10.1002/uog.17347). This source provides gestational-age-specific EFW standards and centile equations. It is appropriate only if the project computes EFW from the required inputs and uses the published calculation; it does not convert an AC-only measurement into EFW or an AC percentile.

### 3. Ramanujan ellipse-perimeter approximation

The original source is S. Ramanujan, “Modular equations and approximations to π,” *Quarterly Journal of Mathematics*, vol. 45, 1914, pp. 350–372. Section 16 gives the empirical ellipse-perimeter approximations, including the first formula used by the project. [Original paper scan](https://ramanujan.sirinudi.org/Volumes/published/ram06.pdf). Cite the original paper for the formula; use the implemented semi-axis convention and formula exactly. It is a geometric approximation, not a clinical validation reference.

### 4. Synthetic-data privacy and DCR

For an empirical health-data privacy attack, cite Zhang Z, Yan C, Malin BA. Membership inference attacks against synthetic health data. *Journal of Biomedical Informatics*. 2022;125:103977. DOI: [10.1016/j.jbi.2021.103977](https://doi.org/10.1016/j.jbi.2021.103977). It demonstrates why synthetic data should not be presumed private and describes membership-inference evaluation.

For a review of privacy and utility metrics in medical synthetic data, cite Kaabachi B, Despraz J, Meurers T, et al. A scoping review of privacy and utility metrics in medical synthetic data. *npj Digital Medicine*. 2025;8:60. DOI: [10.1038/s41746-024-01359-3](https://doi.org/10.1038/s41746-024-01359-3). For a current evaluation framework covering DCR/NNDR and attack-based measures, also see “Evaluating the quality of tabular synthetic data in health care,” *PLOS Digital Health* (2026), DOI: [10.1371/journal.pdig.0001522](https://doi.org/10.1371/journal.pdig.0001522). Treat DCR as a distance-based risk indicator, not a privacy guarantee; pair it with a stated threat model and attack-based assessment where feasible.

## AC calibration note for this project

The ACOUSLIC-AI value of approximately **0.28 mm/pixel** is the image pixel spacing used to convert a pixel-space contour to millimetres. It is not the model’s AC error or evidence of measurement accuracy. Report calibration as mm/pixel and evaluate accuracy separately against reference AC, for example with MAE in mm, bias, and limits of agreement on held-out cases. The user-provided ACOUSLIC-AI v1.1 dataset DOI remains `10.5281/zenodo.12697994`.

## Citation additions ready to merge into the master list

43. Choi HY, Kim JH, Park JY, Jung EY, No JH, Oh KJ, Hong JS. Simple mathematical formulae for estimation of median values of fetal biometry at each gestational age. *Obstet Gynecol Sci*. 2016;59(2):91–96. DOI: 10.5468/ogs.2016.59.2.91.
44. Jung SI, Lee YH, Moon MH, et al. Reference charts and equations of Korean fetal biometry. *Prenat Diagn*. 2007;27(6):545–551. DOI: 10.1002/pd.1729.
45. Kwon JY, Park IY, Wie JH, Choe S, Kim CJ, Shin JC. Fetal biometry in the Korean population: reference charts and comparison with charts from other populations. *Prenat Diagn*. 2014;34(10):927–934. DOI: 10.1002/pd.4394.
46. Stirnemann J, Villar J, Salomon LJ, et al. International estimated fetal weight standards of the INTERGROWTH-21st Project. *Ultrasound Obstet Gynecol*. 2017;49(4):478–486. DOI: 10.1002/uog.17347.
47. Ramanujan S. Modular equations and approximations to π. *Q J Math*. 1914;45:350–372. [Original scan](https://ramanujan.sirinudi.org/Volumes/published/ram06.pdf).
48. Zhang Z, Yan C, Malin BA. Membership inference attacks against synthetic health data. *J Biomed Inform*. 2022;125:103977. DOI: 10.1016/j.jbi.2021.103977.
49. Kaabachi B, Despraz J, Meurers T, et al. A scoping review of privacy and utility metrics in medical synthetic data. *npj Digit Med*. 2025;8:60. DOI: 10.1038/s41746-024-01359-3.

The curated survey retains only references tied to a project module, implemented method, or evaluation claim. The unverified NICE page load remains the sole outstanding item from the URL check.
