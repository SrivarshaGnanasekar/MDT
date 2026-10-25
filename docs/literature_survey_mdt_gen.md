# Focused literature survey and required references for MDT-GEN

**Status:** Curated for the code and outputs currently present in this project. This replaces the broader exploratory list in this file and keeps the sources needed to support implemented modules, their methodological basis, or evaluation claims. The manuscript files were not consulted.

## Literature survey

MDT-GEN currently combines early-pregnancy GDM risk prediction, macrosomia risk estimation, fetal abdominal ultrasound segmentation and biometry, maternal trajectory simulation, and CTGAN-based tabular synthesis. These components have distinct evidence bases and should be validated separately.

For GDM, the IADPSG consensus and HAPO study anchor the diagnostic and clinical context. Recent ML prediction studies provide comparison points, while the 2026 systematic review of externally validated models supports careful reporting of calibration and validation. The project's estimates should be described as model outputs for research unless prospective external validation and clinical utility have been established. The macrosomia studies are comparators, not proof that this project's risk estimates generalize.

For fetal ultrasound, ACOUSLIC-AI supplies annotated sweeps, reference measurements, and the challenge context. U-Net and Attention U-Net support the segmentation architecture; Jang et al. is a direct AC measurement precedent. The project uses 0.28 mm/pixel as ACOUSLIC-AI calibration in the dedicated model path. That number converts pixels to millimetres; it does not measure AC accuracy. Accuracy must be reported separately against reference AC in millimetres. The ellipse perimeter uses Ramanujan's approximation.

The gestational-age median AC equation implemented in `fetal/ac_predict.py`, `AC(cm) = 1.013 × GA(weeks) − 4.634`, matches Choi et al. (2016). That paper derived the regression from reference values after 20 weeks and evaluated it in a Korean cohort. It is a rough median approximation, not a centile curve or a universally validated population standard. The code comment saying the formula applies from 14 weeks exceeds the source's evidence; references alone do not validate that extension.

The fetal-weight path contains AC-only Hadlock and Campbell–Wilkin equations. Cite those original methods and distinguish these AC-only estimates from multi-parameter Hadlock equations. EFW estimates made from a single circumference should not be presented as equivalent to a multi-biometry clinical EFW.

The maternal digital-twin path uses a Neural ODE for continuous-time trajectories, while the CTGAN module generates synthetic tabular records. CTGAN's presence does not guarantee privacy. The project evaluates fidelity, utility, and distance-based privacy indicators; references on membership inference and synthetic-data privacy explain why DCR-style proximity checks are only one part of a privacy assessment.

For prediction-model reporting, retain TRIPOD+AI, PROBAST+AI, calibration, Brier score, and decision-curve analysis sources because the project reports these types of metrics or uses them to frame model validity. Cite the actual implemented estimators and SHAP for the model stack and explanations.

## Required references

### Clinical context and prediction

1. International Association of Diabetes and Pregnancy Study Groups Consensus Panel. International association of diabetes and pregnancy study groups recommendations on the diagnosis and classification of hyperglycemia in pregnancy. *Diabetes Care*. 2010;33(3):676–682. DOI: [10.2337/dc09-1848](https://doi.org/10.2337/dc09-1848).
2. HAPO Study Cooperative Research Group. Hyperglycemia and adverse pregnancy outcomes. *New England Journal of Medicine*. 2008;358:1991–2002. DOI: [10.1056/NEJMoa0707943](https://doi.org/10.1056/NEJMoa0707943).
3. American Diabetes Association Professional Practice Committee. Management of diabetes in pregnancy: Standards of Care in Diabetes—2026. *Diabetes Care*. 2026;49(Suppl 1):S321–S338. DOI: [10.2337/dc26-S015](https://doi.org/10.2337/dc26-S015).
4. Zaky H, Fthenou E, Srour L, et al. Machine learning based model for the early detection of gestational diabetes mellitus. *BMC Medical Informatics and Decision Making*. 2025;25:130. DOI: [10.1186/s12911-025-02947-3](https://doi.org/10.1186/s12911-025-02947-3).
5. Seifu BL, Belsti Y, Yimer NB, Goldstein RF, Enticott J, Teede H. Externally validated risk prediction models for gestational diabetes mellitus: a systematic review and meta-analysis. *Acta Obstetricia et Gynecologica Scandinavica*. 2026;105:1918–1931. DOI: [10.1111/aogs.70300](https://doi.org/10.1111/aogs.70300).
6. Liu Q, Zhu S, Zhao M, et al. Machine learning approaches for predicting fetal macrosomia at different stages of pregnancy: a retrospective study in China. *BMC Pregnancy and Childbirth*. 2025;25:140. DOI: [10.1186/s12884-025-07239-2](https://doi.org/10.1186/s12884-025-07239-2).
7. Wang J, Wang X, Yu Y, Xiang W, Li Y, Zhang X, Sun Q, He Y, Li J, Yuan H. Development and validation of an explainable machine learning-based risk prediction model for macrosomia in Han Chinese women with singleton cephalic pregnancies. *Scientific Reports*. 2026, online ahead of print. DOI: [10.1038/s41598-026-65628-9](https://doi.org/10.1038/s41598-026-65628-9).

### Digital twins and longitudinal modeling

8. Björnsson B, Borrebaeck C, Elander N, et al. Digital twins to personalize medicine. *Genome Medicine*. 2020;12:4. DOI: [10.1186/s13073-019-0701-3](https://doi.org/10.1186/s13073-019-0701-3).
9. Katsoulakis E, Wang Q, Wu H, et al. Digital twins for health: a scoping review. *npj Digital Medicine*. 2024;7:77. DOI: [10.1038/s41746-024-01073-0](https://doi.org/10.1038/s41746-024-01073-0).
10. Chen RTQ, Rubanova Y, Bettencourt J, Duvenaud D. Neural ordinary differential equations. In: *Advances in Neural Information Processing Systems 31*. 2018:6571–6583. [Proceedings record](https://proceedings.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html).

### Fetal growth, AC measurement, and EFW

11. Papageorghiou AT, Ohuma EO, Altman DG, et al. International standards for fetal growth based on serial ultrasound measurements: the Fetal Growth Longitudinal Study of the INTERGROWTH-21st Project. *The Lancet*. 2014;384(9946):869–879. DOI: [10.1016/S0140-6736(14)61490-2](https://doi.org/10.1016/S0140-6736(14)61490-2).
12. Stirnemann J, Villar J, Salomon LJ, et al. International estimated fetal weight standards of the INTERGROWTH-21st Project. *Ultrasound in Obstetrics & Gynecology*. 2017;49(4):478–486. DOI: [10.1002/uog.17347](https://doi.org/10.1002/uog.17347).
13. Salomon LJ, Alfirevic Z, Da Silva Costa F, et al. ISUOG Practice Guidelines: ultrasound assessment of fetal biometry and growth. *Ultrasound in Obstetrics & Gynecology*. 2019;53(6):715–723. DOI: [10.1002/uog.20272](https://doi.org/10.1002/uog.20272).
14. Choi HY, Kim JH, Park JY, Jung EY, No JH, Oh KJ, Hong JS. Simple mathematical formulae for estimation of median values of fetal biometry at each gestational age. *Obstetrics & Gynecology Science*. 2016;59(2):91–96. DOI: [10.5468/ogs.2016.59.2.91](https://doi.org/10.5468/ogs.2016.59.2.91).
15. Sarris I, Ioannou C, Chamberlain P, Ohuma E, Roseman F, Hoch L, Altman DG, Papageorghiou AT, for the International Fetal and Newborn Growth Consortium for the 21st Century. Intra- and interobserver variability in fetal ultrasound measurements. *Ultrasound in Obstetrics & Gynecology*. 2012;39(3):266–273. DOI: [10.1002/uog.10082](https://doi.org/10.1002/uog.10082).
16. Hadlock FP, Harrist RB, Carpenter RJ, Deter RL, Park SK. Sonographic estimation of fetal weight: the value of femur length in addition to head and abdomen measurements. *Radiology*. 1984;150(2):535–540. DOI: [10.1148/radiology.150.2.6691115](https://doi.org/10.1148/radiology.150.2.6691115).
17. Campbell S, Wilkin D. Ultrasonic measurement of fetal abdomen circumference in the estimation of fetal weight. *British Journal of Obstetrics and Gynaecology*. 1975;82(9):689–697. DOI: [10.1111/j.1471-0528.1975.tb00708.x](https://doi.org/10.1111/j.1471-0528.1975.tb00708.x).
18. Sappia MS, de Korte CL, van Ginneken B, et al. ACOUSLIC-AI challenge report: fetal abdominal circumference measurement on blind-sweep ultrasound data from low-income countries. *Medical Image Analysis*. 2025;105:103640. DOI: [10.1016/j.media.2025.103640](https://doi.org/10.1016/j.media.2025.103640).
19. Sappia MS. ACOUSLIC-AI: Abdominal Circumference Operator-agnostic UltraSound measurement in Low-Income Countries using Artificial Intelligence. Dataset v1.1. Zenodo; 2024. DOI: [10.5281/zenodo.12697994](https://doi.org/10.5281/zenodo.12697994).
20. Jang J, Park Y, Kim B, Lee SM, Kwon J-Y, Seo JK. Automatic estimation of fetal abdominal circumference from ultrasound images. *IEEE Journal of Biomedical and Health Informatics*. 2018;22(5):1512–1520. DOI: [10.1109/JBHI.2017.2776116](https://doi.org/10.1109/JBHI.2017.2776116).
21. Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. In: *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*. Springer; 2015:234–241. DOI: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28).
22. Oktay O, Schlemper J, Le Folgoc L, et al. Attention U-Net: learning where to look for the pancreas. arXiv:1804.03999; 2018. [arXiv record](https://arxiv.org/abs/1804.03999).
23. Ramanujan S. Modular equations and approximations to π. *Quarterly Journal of Mathematics*. 1914;45:350–372. [Original paper scan](https://ramanujan.sirinudi.org/Volumes/published/ram06.pdf).

### Synthetic data and privacy

24. Xu L, Skoularidou M, Cuesta-Infante A, Veeramachaneni K. Modeling tabular data using conditional GAN. In: *Advances in Neural Information Processing Systems 32*. 2019:7335–7345. [Proceedings record](https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html).
25. Zhang Z, Yan C, Malin BA. Membership inference attacks against synthetic health data. *Journal of Biomedical Informatics*. 2022;125:103977. DOI: [10.1016/j.jbi.2021.103977](https://doi.org/10.1016/j.jbi.2021.103977).
26. Kaabachi B, Despraz J, Meurers T, et al. A scoping review of privacy and utility metrics in medical synthetic data. *npj Digital Medicine*. 2025;8:60. DOI: [10.1038/s41746-024-01359-3](https://doi.org/10.1038/s41746-024-01359-3).

### Prediction methods, interpretability, and evaluation

27. Breiman L. Random forests. *Machine Learning*. 2001;45(1):5–32. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).
28. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016:785–794. DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
29. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: unbiased boosting with categorical features. In: *Advances in Neural Information Processing Systems 31*. 2018:6638–6648. [Proceedings record](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
30. Wolpert DH. Stacked generalization. *Neural Networks*. 1992;5(2):241–259. DOI: [10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1).
31. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. In: *Advances in Neural Information Processing Systems 30*. 2017:4765–4774. [Proceedings record](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).
32. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378. DOI: [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378).
33. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. *BMJ*. 2025;388:e082505. DOI: [10.1136/bmj-2024-082505](https://doi.org/10.1136/bmj-2024-082505).
34. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Medicine*. 2019;17:230. DOI: [10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7).
35. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making*. 2006;26(6):565–574. DOI: [10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361).
36. Brier GW. Verification of forecasts expressed in terms of probability. *Monthly Weather Review*. 1950;78(1):1–3. DOI: [10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001%3AVOFEIT%3E2.0.CO%3B2).

## Deliberately excluded

The supplied broad list includes background or adjacent work that is not needed to support the current implementation (for example medGAN, HALO, Latent ODE/CDE methods, and unrelated digitization or clinical-AI papers). They can be added back only if the survey discusses or the project implements those methods. NICE NG3 is omitted from this focused set because the ADA Standards source already covers the current diabetes-in-pregnancy guidance needed here; NICE's URL remains an official link but could not be loaded by the checker.


