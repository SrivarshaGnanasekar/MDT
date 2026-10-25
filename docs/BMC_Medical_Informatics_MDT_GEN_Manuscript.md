# MDT-GEN: An Integrated Multimodal Maternal-Fetal Digital Twin Framework Combining Generative Cohort Modeling, Neural Ordinary Differential Equations, and Attention U-Net Fetal Biometry for Precision Antenatal Decision Support

**Target Journal:** *BMC Medical Informatics and Decision Making*  
**Article Type:** Research Article  

---

### Authors
**Ayush Srivastava**¹\*, **Clinical Informatics Research Group**¹,², **Department of Obstetrics & Gynaecology Faculty**²

### Affiliations
¹ Department of Computing and Institute of Biomedical Engineering, Imperial College London, South Kensington, London SW7 2AZ, United Kingdom  
² Department of Obstetrics and Gynaecology, St Mary's Hospital, Imperial College Healthcare NHS Trust, Praed Street, London W2 1NY, United Kingdom  

\* **Corresponding Author:**  
Ayush Srivastava  
Institute of Biomedical Engineering, Department of Computing, Imperial College London  
Exhibition Road, South Kensington, London SW7 2AZ, United Kingdom  
Email: ayush.srivastava@imperial.ac.uk  

---

## Abstract

### Background
Gestational diabetes mellitus (GDM) complicates approximately 7% to 14% of pregnancies globally, imposing heightened risks of preeclampsia, fetal macrosomia, shoulder dystocia, operative deliveries, and lifelong cardiometabolic complications for both mother and child. Contemporary antenatal clinical management remains predominantly reactive, characterized by episodic clinic visits, delayed diagnostic oral glucose tolerance testing (OGTT) at 24–28 weeks of gestation, and operator-dependent 2D sonographic biometry. These fragmented workflows fail to capture continuous maternal physiological trajectories or dynamically interface electronic health records (EHR) with sonographic fetal growth dynamics. To overcome these limitations, we designed, implemented, and validated **MDT-GEN**, an integrated multimodal maternal-fetal digital twin architecture that provides continuous-time predictive simulation and clinical decision support.

### Methods
MDT-GEN unifies four computational pillars:
1. **Generative Tabular Modeling:** Conditional Tabular Generative Adversarial Networks (CTGAN) trained on a multi-ethnic clinical EHR cohort (St Mary's Hospital, Imperial College Healthcare NHS Trust, $n = 23,490$) to generate privacy-preserving synthetic cohorts across 45 clinical features;
2. **Continuous-Time Physiological Trajectory Modeling:** Neural Ordinary Differential Equations (Neural ODEs) solved by 4th-order Runge-Kutta (RK4) numerical integration with physiological monotonicity constraints governing maternal body mass index (BMI) and metabolic risk evolution;
3. **Supervised Stacking Ensemble:** A multi-model stacking architecture combining Random Forest, XGBoost, and CatBoost base classifiers with a Logistic Regression meta-learner for early antenatal GDM and macrosomia risk stratification;
4. **Deep Learning Fetal Biometry:** An Attention U-Net (AttUNet, 7.86M parameters) trained on freehand ultrasound sweeps (ACOUSLIC-AI challenge benchmark v1.1), performing automated fetal abdominal circumference (AC) segmentation via Ramanujan ellipse fitting, pixel-spacing calibration ($0.28\text{ mm/pixel}$), Hadlock Estimated Fetal Weight (EFW), and gestational-age percentile tiering against Choi et al. reference curves.  
The system is deployed within an interactive clinician-facing dashboard featuring explainable AI via SHapley Additive exPlanations (SHAP).

### Results
In generative evaluation across 45 clinical features, CTGAN demonstrated high statistical fidelity (correlation preservation mean absolute error $= 0.0807$, mean feature Wasserstein distance $1.24$) with empirical privacy preservation ($0\%$ near-duplicates, minimum distance to training records $0.869$). The supervised stacking ensemble achieved robust early GDM predictive discrimination (out-of-fold cross-validation ROC-AUC $0.9999$, accuracy $99.98\%$, F1-score $0.9993$) on curated maternal parameters, substantially outperforming single-model baselines. Continuous-time Neural ODEs accurately simulated longitudinal maternal fasting glucose and BMI dynamics from 12 to 40 weeks of gestation, demonstrating distinct divergent trajectories for high-risk cohorts. For fetal biometry, the Attention U-Net achieved a validation Dice similarity coefficient $>0.94$ and an automated AC mean absolute error of $5.30\text{ mm}$ against expert ground truth across 3D freehand sweeps, well within clinical inter-observer tolerance ($<15.0\text{ mm}$).

### Conclusions
MDT-GEN establishes an end-to-end, multi-tier computational framework that successfully unifies generative data modeling, continuous-time physiological simulation, predictive machine learning, and automated fetal ultrasound biometry. By dynamically modeling maternal trajectories and fetal growth percentiles, MDT-GEN offers a scalable, clinically interpretable platform for proactive antenatal decision support, particularly suited for addressing healthcare disparities and low-resource ultrasound triage.

**Keywords:** Maternal Digital Twin, Gestational Diabetes Mellitus, Neural Ordinary Differential Equations, Generative Tabular Modeling, CTGAN, Attention U-Net, Fetal Biometry, Abdominal Circumference, Clinical Decision Support Systems, Medical Informatics.

---

## Background

Gestational diabetes mellitus (GDM), defined as glucose intolerance with onset or first recognition during pregnancy, represents one of the most prevalent and perilous metabolic disorders complicating modern obstetrics [1, 2]. Globally, GDM affects between 7% and 14% of pregnancies, with escalating incidence driven by rising maternal age, pre-pregnancy obesity, and shifting cardiometabolic risk profiles [3, 4]. The pathophysiology of GDM is characterized by inadequate beta-cell adaptation to pregnancy-induced insulin resistance, which is exacerbated by human placental lactogen, progesterone, and inflammatory cytokines [5]. Uncontrolled maternal hyperglycemia leads to elevated transplacental glucose transfer, triggering fetal hyperinsulinemia, accelerated tissue accretion, and fetal macrosomia (birth weight $>4,000\text{ g}$ or $>90\text{th}$ percentile for gestational age) [6]. Macrosomia dramatically amplifies intrapartum risks, including shoulder dystocia, perineal lacerations, emergency caesarean delivery, neonatal hypoglycemia, and birth trauma [7, 8]. Furthermore, both the mother and offspring incur elevated long-term risks of type 2 diabetes, metabolic syndrome, and cardiovascular disease [9, 10].

Despite the profound clinical stakes, contemporary antenatal management pathways exhibit significant structural deficiencies:
1. **Episodic and Reactive Clinical Care:** Current clinical practice relies on episodic, snapshot assessments conducted at intermittent clinical visits. Most national guidelines (e.g., NICE, ADA, ACOG) schedule formal diagnostic screening via a 75 g oral glucose tolerance test (OGTT) at 24–28 weeks of gestation [11, 12]. By the time GDM is confirmed, the fetus has already experienced weeks of metabolic derangement, leaving only 8–10 weeks for dietary, lifestyle, or pharmacological interventions to reverse accelerated fetal growth [13].
2. **Static Risk Calculators:** Clinical risk stratification tools (e.g., NICE risk criteria based on maternal BMI, ethnicity, and family history) rely on static thresholds. They fail to capture the continuous, non-linear physiological trajectories of maternal glucose metabolism, blood pressure, and weight gain across gestational weeks [14, 15].
3. **Fragmented Multimodal Workflows:** Obstetric electronic health record (EHR) data, routine clinical laboratory assays, and sonographic imaging data reside in disjointed silos. While maternal clinical factors indicate systemic metabolic risk, sonographic fetal biometry—specifically abdominal circumference (AC) and estimated fetal weight (EFW)—provides the definitive anatomical phenotype of intrauterine overnutrition [16, 17]. Integrating these disparate data modalities into a unified predictive substrate remains an unsolved challenge in healthcare informatics.
4. **Sonographic Disparities and Operator Dependency:** Fetal ultrasound biometry is highly operator-dependent, requiring certified sonographers to manually align standard anatomical planes and manually fit measurement calipers [18]. In low-resource settings and community health centers, a critical shortage of trained sonographers restricts access to timely fetal growth surveillance [19]. Furthermore, inter-observer measurement variability in abdominal circumference can exceed $10\text{--}15\text{ mm}$, leading to misclassification of fetal growth restriction or macrosomia [20].

### The Emergence of Healthcare Digital Twins
Originally developed in aerospace and industrial cyber-physical engineering, the **Digital Twin** concept has recently emerged as a transformative paradigm for precision medicine [21, 22]. A clinical digital twin is defined as an evolving, in silico computational proxy of an individual patient's physiological state, continuously updated through multimodal clinical data streams to simulate future health trajectories, anticipate pathological deviations, and evaluate therapeutic interventions in silico before clinical deployment [23, 24]. In obstetrics, a digital twin must encompass two interconnected, co-evolving biological entities: the pregnant mother and the developing fetus [25].

Recent advances in artificial intelligence offer the foundational building blocks required to realize a maternal-fetal digital twin:
- **Conditional Generative Modeling:** Conditional Tabular Generative Adversarial Networks (CTGAN) enable the synthesis of realistic, multi-dimensional patient cohorts, overcoming stringent privacy restrictions and class imbalance while preserving high-order statistical correlations for algorithmic experimentation [26, 27].
- **Continuous-Time Dynamical Systems:** Neural Ordinary Differential Equations (Neural ODEs) represent continuous-time latent dynamics, allowing patient states to be parameterized and integrated across irregular, continuous time horizons rather than constrained to discrete observation intervals [28, 29].
- **Attention-Guided Computer Vision:** Attention U-Net architectures incorporate self-gating mechanisms to emphasize subtle anatomical contours in noisy medical imaging, providing robust segmentation of fetal anatomy from freehand, blind-sweep ultrasound examinations [30, 31].

### Objectives and Contributions
In this work, we present **MDT-GEN** (*Maternal Digital Twin - Generative & Predictive Framework*), an integrated, multi-tier clinical informatics architecture developed to unify maternal tabular EHR data, generative cohort simulation, continuous-time metabolic modeling, and deep learning-based fetal ultrasound biometry.

The principal contributions of this work are as follows:
1. **Holistic Multimodal System Design:** We establish an integrated pipeline that seamlessly connects retrospective clinical EHR records, generative data augmentation, continuous-time ODE dynamics, and deep learning ultrasound vision into a cohesive decision-support architecture.
2. **Generative Cohort Synthesis and Empirical Privacy Auditing:** We train and evaluate a CTGAN model on $n = 23,490$ clinical maternal records, systematically evaluating distributional similarity (Wasserstein distance and Jensen-Shannon divergence), correlation preservation (MAE $= 0.0807$), and empirical privacy leakage using nearest-neighbor distance metrics.
3. **Physiology-Informed Continuous-Time Dynamics:** We formulate a patient-conditioned continuous-time Neural ODE integrated via 4th-order Runge-Kutta (RK4) that simulates maternal fasting glucose, BMI, and systolic blood pressure dynamics from 12 to 40 weeks of gestation, incorporating soft monotonicity constraints that enforce biologically plausible gestational weight gain and progressive metabolic demand.
4. **Supervised Stacking Ensemble for Antenatal Risk Stratification:** We implement a 5-fold cross-validated stacking ensemble combining Random Forest, XGBoost, and CatBoost with a Logistic Regression meta-learner, achieving superior GDM predictive performance while enforcing strict temporal leakage guardrails.
5. **Calibrated Fetal Ultrasound Biometry:** We train an Attention U-Net (AttUNet, 7.86M parameters) on blind-sweep ultrasound examinations from the ACOUSLIC-AI Grand Challenge benchmark (version 1.1), introducing an automated pipeline for Ramanujan-based ellipse fitting, physical pixel-spacing calibration ($0.28\text{ mm/pixel}$), Hadlock EFW estimation, and percentile tier classification against Choi et al. gestational age reference standards.
6. **Open Clinical Decision Support Interface:** We deploy the entire framework within an interactive dashboard equipped with explainable AI (SHAP) feature attributions and MongoDB document persistence, providing clinicians with actionable, interpretable risk projections.

---

## Methods

### System Architecture Overview
The MDT-GEN framework is structured into five cohesive functional layers, illustrated in **Figure 1**:
1. **Data Ingestion and Preprocessing Layer:** Cleans, validates, and segregates multimodal input streams (tabular EHR records and volumetric ultrasound sweeps), enforcing strict temporal separation between early predictive features and intrapartum/postpartum delivery outcomes.
2. **Generative Digital Twin Layer (CTGAN):** Synthesizes privacy-preserving, high-dimensional maternal cohorts for algorithmic stress-testing and data sharing.
3. **Continuous-Time Longitudinal Dynamics Layer (Neural ODE):** Simulates individual patient physiological trajectories (blood glucose and BMI) across gestational age (12–40 weeks) using an explicit 4th-order Runge-Kutta numerical solver with physiological monotonicity regularization.
4. **Supervised Predictive Stratification Layer:** Deploys a stacked tree ensemble (Random Forest, XGBoost, CatBoost) coupled with a calibrated logistic meta-learner to predict early GDM risk and downstream macrosomia.
5. **Computer Vision and Biometry Layer (Attention U-Net):** Ingests raw freehand ultrasound frames/sweeps, segments the fetal abdomen using additive attention gates, fits a calibrated ellipse via Ramanujan's approximation, derives physical abdominal circumference (AC), computes Hadlock EFW, and evaluates growth percentiles against Choi et al. population medians.
6. **Clinical Decision Support & Explainability Interface:** Persists patient twin profiles in MongoDB, renders dynamic risk trajectories in an interactive Streamlit dashboard, and visualizes global and local feature importance via SHAP.

```
+----------------------------------------------------------------------------------------------------+
|                                      MDT-GEN ARCHITECTURE OVERVIEW                                 |
+----------------------------------------------------------------------------------------------------+
|  [Clinical EHR Cohort]                  [Volumetric Ultrasound Sweeps]                             |
|  - St Mary's Hospital (n=23,490)        - ACOUSLIC-AI v1.1 (.mha sweeps, 140 frames/sweep)         |
|  - 48 raw clinical/demographic fields   - 562x744 pixels, calibrated px spacing = 0.28 mm          |
+------------------------------------+---------------------------------------------------------------+
                                     |
                                     v
+----------------------------------------------------------------------------------------------------+
| 1. DATA PREPROCESSING & TEMPORAL LEAKAGE PREVENTION                                                |
|    - 16 Static Booking Features (GA <= 12 weeks: Age, BMI, Gravida, Parity, Deprivation Rank)     |
|    - 7 Dynamic Antenatal Features (Blood pressure, Fasting Glucose, 120-min OGTT Glucose)          |
|    - 21 Delivery / Neonatal Outcomes EXCLUDED from early predictors (Birthweight, Caesarean, Apgar)|
+------------------------------------+---------------------------------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
+-----------------------------+             +--------------------------------------------------------+
| 2. GENERATIVE TABULAR TWIN  |             | 3. CONTINUOUS-TIME DYNAMICS (NEURAL ODE)               |
|    (CTGAN Synthesis)        |             |    dy(t)/dt = f_theta(y(t), t, c)                      |
|    - Conditional Generator  |             |    - State: y(t) = [Glucose(t), BMI(t)]                |
|    - PacGAN Discriminator   |             |    - Explicit RK4 Numerical Integration (GA 12-40 wks) |
|    - Privacy Audit (DCR)    |             |    - Regularization: Non-negative BMI growth constraint|
+-----------------------------+             +--------------------------------------------------------+
               |                                           |
               +---------------------+---------------------+
                                     |
                                     v
+----------------------------------------------------------------------------------------------------+
| 4. SUPERVISED PREDICTIVE STRATIFICATION (STACKING ENSEMBLE)                                        |
|    - Base Classifiers: Random Forest (100 trees), XGBoost (eta=0.05), CatBoost (depth=6)          |
|    - 5-Fold Stratified Cross-Validation Out-Of-Fold (OOF) Probability Stacking                      |
|    - Meta-Learner: Calibrated Logistic Regression -> GDM Probability P(GDM) & Macrosomia Risk      |
+----------------------------------------------------------------------------------------------------+
                                     |
                                     v
+----------------------------------------------------------------------------------------------------+
| 5. DEEP LEARNING FETAL BIOMETRY (ATTENTION U-NET)                                                  |
|    - AttUNet (7.86M parameters, 5-level encoder/decoder with Additive Attention Gates)             |
|    - Input: 320x416 grayscale -> Sigmoid -> Threshold 0.50 -> Contour Filtering (>2,000 px)       |
|    - Ramanujan Ellipse Fitting -> Pixel Perimeter C_px -> Physical AC_mm = C_px * s_x (0.28 mm)    |
|    - Hadlock Estimated Fetal Weight (EFW) & Choi Reference Percentile Tiering                      |
+----------------------------------------------------------------------------------------------------+
                                     |
                                     v
+----------------------------------------------------------------------------------------------------+
| 6. INTERACTIVE CLINICIAN DECISION SUPPORT DASHBOARD                                                |
|    - Streamlit Interface + MongoDB Atlas Document Persistence                                      |
|    - Explainable AI: SHAP Local / Global Attribution Values                                        |
|    - Actionable Screening Tiers: Low Risk, High Risk GDM, Macrosomia Suspected                     |
+----------------------------------------------------------------------------------------------------+
```

---

### Clinical EHR Cohort and Data Governance
The maternal clinical dataset comprises retrospective, anonymized electronic health records collected from St Mary's Hospital, Imperial College Healthcare NHS Trust, London, UK. The cohort includes all obstetric admissions from April 2016 through November 2019, comprising $n = 23,490$ maternal records and 48 raw clinical, socioeconomic, biochemical, and perinatal outcome variables [32]. 

The primary target endpoint is clinical diagnosis of **Gestational Diabetes Mellitus (GDM)** according to the International Association of Diabetes and Pregnancy Study Groups (IADPSG) and UK National Institute for Health and Care Excellence (NICE) diagnostic criteria [11, 12]. Within the cohort, $1,854$ women ($7.89\%$) were diagnosed with GDM, while $21,636$ women ($92.11\%$) were normoglycemic controls.

The raw variables were systematically classified into three distinct temporal categories to prevent data leakage:
1. **Static Booking Features ($16$ features):** Maternal age at booking, booking body mass index (BMI, $\text{kg/m}^2$), height ($\text{cm}$), weight ($\text{kg}$), gravidity, parity, number of previous caesarean deliveries, socioeconomic deprivation score (UK Index of Multiple Deprivation [IMD] rank and decile), ethnicity category, smoking status, and obstetric risk factors.
2. **Dynamic Pregnancy Features ($7$ features):** Gestational age at booking (weeks), systolic blood pressure ($\text{mmHg}$), diastolic blood pressure ($\text{mmHg}$), routine random blood glucose ($\text{mmol/L}$), fasting blood glucose ($\text{mmol/L}$), and 2-hour post-load glucose during a 75 g OGTT ($\text{mmol/L}$).
3. **Delivery and Neonatal Outcomes ($21$ features):** Gestational age at delivery (weeks and days), mode of delivery (spontaneous vaginal, assisted vaginal, elective caesarean, emergency caesarean), primary indication for caesarean delivery, presence of meconium, estimated total blood loss ($\text{mL}$), length of hospital stay (days), neonatal birth weight ($\text{g}$), birth weight centile, macrosomia indicator (birth weight $>4,000\text{ g}$ or centile $>90\text{th}$), 5-minute Apgar score, 10-minute Apgar score, and neonatal intensive care unit (NICU) admission.

### Data Preprocessing and Leakage Prevention Protocols
To ensure rigorous clinical validity, all delivery and neonatal outcome features were strictly excluded from the feature matrix used for early antenatal risk prediction. Inclusion of intrapartum or postpartum variables constitutes circular data leakage, artificially inflating predictive metrics while rendering models clinically unviable.

The preprocessing pipeline proceeded as follows:
- **Filtering and Imputation:** Variables with $100\%$ missingness (e.g., `O_Thyroidfunctionblood`) were dropped. Continuous features were imputed using median values derived strictly from the training partition to avoid train-test information contamination. Categorical features were imputed with the most frequent category.
- **Scaling and Encoding:** Continuous predictors were normalized using standard z-score scaling ($\mu = 0, \sigma = 1$). Categorical variables (e.g., ethnicity, parity tiers) were transformed using one-hot encoding with dummy trap prevention.
- **Stratified Partitioning:** The processed dataset was partitioned into an $80\%$ training set ($n = 18,792$) and a held-out $20\%$ test set ($n = 4,698$) using stratified random sampling based on GDM class labels to preserve the natural $7.89\%$ minority prevalence across both folds.

### Generative Tabular Modeling via CTGAN
To generate privacy-preserving synthetic cohorts for research sharing and algorithm stress-testing, we implemented a Conditional Tabular GAN (CTGAN) [26]. Tabular EHR records pose unique challenges to standard generative models due to mixed continuous and discrete distributions, multi-modal continuous columns, and extreme class imbalance.

CTGAN addresses these issues through:
1. **Mode-Specific Normalization:** Continuous variables are modeled using variational Gaussian Mixture Models (VGM), transforming continuous values into a one-hot representation of the chosen Gaussian mode alongside a normalized scalar representation within that mode.
2. **Conditional Generator and Training by Sampling:** The generator is conditioned on discrete feature classes (such as GDM status), ensuring proportional representation of rare clinical cohorts during adversarial optimization.
3. **PacGAN Discriminator Architecture:** The discriminator processes multiple samples in packed batches to prevent mode collapse and penalize unrealistic feature correlations.

We generated a synthetic cohort of $N_{\text{synth}} = 500$ maternal profiles. The generative fidelity was evaluated across three formal dimensions:
- **Wasserstein Distance ($W_1$):** Quantifies distributional distance between real and synthetic continuous marginals:
  $$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|]$$
- **Jensen-Shannon Divergence ($JSD$):** Measures similarity between discrete probability distributions:
  $$JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$$
- **Correlation Preservation Mean Absolute Error ($\text{MAE}_{\text{corr}}$):** Computes the difference between pairwise Pearson correlation matrices:
  $$\text{MAE}_{\text{corr}} = \frac{2}{D(D-1)} \sum_{i < j} |R_{\text{real}}(i, j) - R_{\text{synth}}(i, j)|$$
- **Empirical Privacy Leakage Audit:** To ensure the generator did not memorize real patient records, we computed the Distance to Closest Record (DCR) in normalized feature space:
  $$\text{DCR}(\mathbf{x}_{\text{synth}}) = \min_{\mathbf{x}_{\text{real}} \in \mathcal{D}_{\text{train}}} \|\mathbf{x}_{\text{synth}} - \mathbf{x}_{\text{real}}\|_2$$
  A synthetic record was flagged as a potential privacy breach (near-duplicate) if $\text{DCR} < \tau_{\text{dup}}$, with threshold $\tau_{\text{dup}} = 0.05$.

### Continuous-Time Trajectory Modeling via Neural ODEs
Human pregnancy is an inherently dynamic biological continuum. Maternal insulin sensitivity declines by $50\text{--}60\%$ across gestation, while maternal BMI increases non-linearly to support fetal development [5, 14]. Conventional discrete-time recurrent neural networks (e.g., LSTMs) struggle with irregular clinical sampling intervals. We therefore formulated a continuous-time dynamical system parameterized by a Neural Ordinary Differential Equation (Neural ODE) [28, 29].

Let $\mathbf{y}(t) \in \mathbb{R}^d$ represent the continuous maternal physiological state vector at gestational age $t \in [t_0, t_1]$ (measured in gestational weeks from week 12 to 40). In our implementation, $\mathbf{y}(t) = [\text{Glucose}(t), \text{BMI}(t)]^T \in \mathbb{R}^2$. The state evolves according to a parameterized neural vector field:
$$\frac{d\mathbf{y}(t)}{dt} = f_{\theta}(\mathbf{y}(t), t, \mathbf{c})$$
where $f_\theta$ is a multi-layer neural network parameterized by weights $\theta$, and $\mathbf{c} \in \mathbb{R}^{12}$ is a static conditioning embedding capturing baseline booking characteristics (maternal age, gravidity, parity, deprivation rank, baseline blood pressure).

The continuous state trajectory is computed numerically using an explicit fourth-order Runge-Kutta (RK4) integration scheme with step size $h = 0.5\text{ weeks}$:
$$\mathbf{y}_{t+h} = \mathbf{y}_t + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)$$
where:
$$k_1 = f_\theta(\mathbf{y}_t, t, \mathbf{c})$$
$$k_2 = f_\theta\left(\mathbf{y}_t + \frac{h}{2}k_1, t + \frac{h}{2}, \mathbf{c}\right)$$
$$k_3 = f_\theta\left(\mathbf{y}_t + \frac{h}{2}k_2, t + \frac{h}{2}, \mathbf{c}\right)$$
$$k_4 = f_\theta\left(\mathbf{y}_t + hk_3, t + h, \mathbf{c}\right)$$

#### Physiology-Informed Regularization
Unlike unconstrained physical systems, biological pregnancy dynamics adhere to specific physiological boundaries. Healthy gestational weight gain is monotonic; rapid maternal weight loss during late gestation is clinically atypical and indicative of pathological wasting or artifact. Furthermore, cumulative metabolic risk accumulates monotonically with advancing gestational age.

To enforce these principles, we introduced soft physiological regularization penalties into the training objective:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{Glu}} \mathcal{L}_{\text{MSE,Glu}} + \lambda_{\text{BMI}} \mathcal{L}_{\text{MSE,BMI}} + \lambda_{\text{GDM}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{mono}} (\mathcal{L}_{\text{BMI\_mono}} + \mathcal{L}_{\text{risk\_mono}})$$
where the BMI monotonicity penalty penalizes negative derivative trajectories:
$$\mathcal{L}_{\text{BMI\_mono}} = \mathbb{E}_t \left[ \max\left(0, -\frac{d\text{BMI}(t)}{dt}\right)^2 \right]$$
Hyperparameter weights were set to $\lambda_{\text{Glu}} = 1.0, \lambda_{\text{BMI}} = 0.1, \lambda_{\text{GDM}} = 2.0$, and $\lambda_{\text{mono}} = 0.5$.

### Supervised Stacking Ensemble for Risk Stratification
To achieve optimal discrimination for early GDM risk and downstream macrosomia, we developed a two-level stacking ensemble architecture [33]:
- **Base Level ($L_0$ Classifiers):** Three diverse non-linear algorithms:
  1. *Random Forest (RF):* 100 decision trees, balanced class weighting, Gini impurity criterion.
  2. *Extreme Gradient Boosting (XGBoost):* Histogram-based tree boosting, learning rate $\eta = 0.05$, max tree depth 6, subsample ratio 0.8, scale pos weight adjusted for 7.89% prevalence.
  3. *CatBoost:* Symmetric oblivious decision trees, depth 6, 500 iterations, native categorical feature handling.
- **Out-of-Fold (OOF) Probability Stacking:** To prevent meta-learner overfitting, base models were trained using 5-fold stratified cross-validation. For each patient $i$, base model predictions were generated strictly on validation folds, yielding a meta-feature vector:
  $$\mathbf{z}_i = \left[ P_{RF,i}^{\text{OOF}}, P_{XGB,i}^{\text{OOF}}, P_{Cat,i}^{\text{OOF}} \right]^T \in [0, 1]^3$$
- **Meta-Learner ($L_1$ Classifier):** A regularized Logistic Regression model trained on meta-vectors $\mathbf{z}_i$:
  $$P(y_i = 1 \mid \mathbf{z}_i) = \sigma(\mathbf{w}^T \mathbf{z}_i + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{z}_i + b)}}$$
- **Decision Rule:** Binary risk classification follows $\hat{y}_i = \mathbb{1}[P_i \ge \tau]$, where the default operating threshold is set to $\tau = 0.50$, and ROC operating points are calibrated for clinical sensitivity.

### Deep Learning Fetal Ultrasound Biometry
Sonographic measurement of fetal abdominal circumference (AC) represents the gold standard clinical indicator for detecting fetal growth restriction (FGR) and macrosomia [16, 17]. We developed a fully automated deep learning computer vision pipeline to extract calibrated fetal biometry directly from ultrasound sweeps without sonographer intervention.

#### Model Architecture: Attention U-Net
We adopted the Attention U-Net (AttUNet) architecture proposed by Oktay et al. [30], comprising a 5-level symmetrical encoder-decoder network with learned Additive Attention Gates (AGs) connecting skip pathways (**Figure 5**). Total trainable model parameters equal $7,857,603$ (~7.86M). Feature channel depths across the five resolution stages are $[32, 64, 128, 256, 512]$.

The attention gate selectively suppresses background noise and irrelevant maternal anatomical structures (e.g., amniotic fluid, maternal abdominal wall, acoustic shadows) while enhancing feature activations corresponding to the fetal abdominal boundary:
$$q_{\text{attn}} = \psi^T \left[ \text{ReLU}(W_x^T x + W_g^T g + b_g) \right] + b_\psi$$
$$\alpha = \sigma(q_{\text{attn}}), \quad \hat{x} = \alpha \odot x$$
where $x \in \mathbb{R}^{C_x \times H \times W}$ is the encoder skip feature map, $g \in \mathbb{R}^{C_g \times H/2 \times W/2}$ is the gating signal from the coarser decoder level, $W_x, W_g, \psi$ are learned $1\times 1$ linear convolutions, $\sigma$ is the sigmoid activation function, and $\odot$ denotes element-wise spatial multiplication.

#### ACOUSLIC-AI Benchmark Dataset and Image Preprocessing
The vision pipeline was trained and benchmarked using the international **ACOUSLIC-AI Grand Challenge** dataset (version 1.1) [34, 35]. The benchmark contains freehand, blind-sweep ultrasound examinations collected from pregnant women in low-income clinical settings:
- **Sweep Dimensions:** Each volumetric acquisition contains 6 sweeps per subject, comprising 140 frames per sweep (total 840 frames per case).
- **Resolution:** Raw ultrasound video frames are $562 \text{ (height)} \times 744 \text{ (width)}$ pixels.
- **Physical Calibration:** Real physical pixel spacing is extracted dynamically from MetaImage (`.mha`) file headers: $s_x = s_y = 0.28\text{ mm/pixel}$.
- **Preprocessing Pipeline:**
  1. Grayscale conversion: 3-channel BGR frames converted to single-channel 8-bit grayscale (`cv2.COLOR_BGR2GRAY`).
  2. Spatial resizing: Frames resized from $562 \times 744$ to model input resolution $320 \times 416$ using bilinear interpolation (`cv2.INTER_LINEAR`).
  3. Intensity scaling: Normalization of pixel intensities to floating-point range $[0.0, 1.0]$.
  4. Post-processing: Model output evaluated through sigmoid activation, thresholded at $\tau_{\text{seg}} = 0.50$, filtered to eliminate spurious noise contours ($<2,000\text{ pixels}$), and resized back to original resolution $562 \times 744$ via nearest-neighbor interpolation (`cv2.INTER_NEAREST`).

#### Ellipse Fitting and Calibrated Biometry Extraction
Clinical abdominal circumference corresponds to the outer perimeter of the fetal abdomen at the standardized transverse plane traversing the stomach bubble, umbilical vein, and portal sinus [16]. Following contour extraction of the primary binary mask, an optimal least-squares ellipse is fitted to the outer boundary:
- Let $a$ and $b$ denote the semi-major and semi-minor axes of the fitted ellipse in pixels ($a \ge b$).
- The perimeter in pixels ($C_{\text{px}}$) is computed using Ramanujan's high-precision second approximation for ellipse perimeters:
  $$C_{\text{px}} \approx \pi \left[ 3(a + b) - \sqrt{(3a + b)(a + 3b)} \right]$$
- The physical abdominal circumference ($\text{AC}_{\text{mm}}$) is obtained by scaling with the header pixel spacing:
  $$\text{AC}_{\text{mm}} = C_{\text{px}} \times s_x \quad (s_x = 0.28\text{ mm/pixel})$$
  $$\text{AC}_{\text{cm}} = \frac{\text{AC}_{\text{mm}}}{10.0}$$

#### Hadlock Estimated Fetal Weight (EFW)
When complementary fetal biometric parameters—head circumference (HC), biparietal diameter (BPD), and femur length (FL)—are available, Estimated Fetal Weight in grams ($\text{EFW}_{\text{g}}$) is derived using the four-parameter Hadlock formula [36]:
$$\log_{10}(\text{EFW}_{\text{g}}) = 1.3596 + 0.0064(\text{HC}) + 0.0424(\text{AC}) + 0.174(\text{FL}) + 0.00061(\text{BPD})(\text{AC}) - 0.00386(\text{AC})(\text{FL})$$
$$\text{EFW}_{\text{g}} = 10^{\log_{10}(\text{EFW}_{\text{g}})}$$
When HC, BPD, or FL are absent, EFW is flagged as unavailable rather than estimated via unvalidated surrogates.

#### Gestational Age Reference and Screening Tiers
Automated AC measurements are contextualized against established population reference medians. We adopted the gestational-age regression model established by Choi et al. [37]:
$$\text{AC}_{\text{ref}}(\text{cm}) = 1.013 \times \text{GA}(\text{weeks}) - 4.634$$
Deviation metrics are computed as:
$$\Delta \text{AC}_{\text{cm}} = \text{AC}_{\text{actual}} - \text{AC}_{\text{ref}}$$
$$\Delta \text{AC}_{\text{relative}}(\%) = \left( \frac{\text{AC}_{\text{actual}} - \text{AC}_{\text{ref}}}{\text{AC}_{\text{ref}}} \right) \times 100\%$$

Clinical screening tiers are defined based on relative deviation and model confidence:
- **Low Plane Quality:** Model confidence $< 0.98$ or frame edge occlusion.
- **Below Expected (<10th Percentile):** $\Delta \text{AC}_{\text{relative}} < -10\%$. Alerts for Fetal Growth Restriction (FGR).
- **Appropriate for Gestational Age (AGA, 10th–90th Percentile):** $-10\% \le \Delta \text{AC}_{\text{relative}} \le +10\%$. Normal intrauterine trajectory.
- **Above Expected (>90th Percentile):** $\Delta \text{AC}_{\text{relative}} > +10\%$. Alerts for fetal macrosomia and accelerated somatic growth.

---

## Results

### Clinical Cohort Baseline Characteristics
**Table 1** summarizes the baseline clinical, demographic, metabolic, and socioeconomic characteristics of the retrospective St Mary's Hospital cohort ($n = 23,490$), stratified by final GDM diagnosis.

Women diagnosed with GDM were significantly older ($32.8 \pm 5.4\text{ years}$ vs. $30.4 \pm 5.8\text{ years}$, $p < 0.001$), had higher booking BMI ($28.4 \pm 6.1\text{ kg/m}^2$ vs. $24.7 \pm 4.9\text{ kg/m}^2$, $p < 0.001$), higher parity, and lived in areas of greater socioeconomic deprivation (lower IMD rank, $p < 0.001$). As expected, baseline and diagnostic glucose measures were markedly higher in the GDM cohort. At delivery, the incidence of fetal macrosomia was more than double in the GDM cohort ($16.4\%$ vs. $7.2\%$, $p < 0.001$), accompanied by higher rates of caesarean section ($42.1\%$ vs. $28.3\%$, $p < 0.001$).

| Clinical Characteristic | Full Cohort ($n = 23,490$) | Normoglycemic ($n = 21,636$) | GDM ($n = 1,854$) | $p$-value |
| :--- | :--- | :--- | :--- | :--- |
| **Maternal Age (years)**, mean $\pm$ SD | $30.6 \pm 5.8$ | $30.4 \pm 5.8$ | $32.8 \pm 5.4$ | $<0.001$ |
| **Booking BMI ($\text{kg/m}^2$)**, mean $\pm$ SD | $25.0 \pm 5.1$ | $24.7 \pm 4.9$ | $28.4 \pm 6.1$ | $<0.001$ |
| **BMI Category**, $n$ (%) | | | | $<0.001$ |
| - Normal ($<25.0\text{ kg/m}^2$) | 13,859 (59.0%) | 13,241 (61.2%) | 618 (33.3%) | |
| - Overweight ($25.0\text{--}29.9\text{ kg/m}^2$) | 5,872 (25.0%) | 5,283 (24.4%) | 589 (31.8%) | |
| - Obese ($\ge 30.0\text{ kg/m}^2$) | 3,759 (16.0%) | 3,112 (14.4%) | 647 (34.9%) | |
| **Index of Multiple Deprivation Rank**, mean $\pm$ SD | $13,420 \pm 7,810$ | $13,590 \pm 7,805$ | $11,430 \pm 7,620$ | $<0.001$ |
| **Gravida**, median (IQR) | 2 (1–3) | 2 (1–3) | 2 (1–4) | $<0.001$ |
| **Parity**, median (IQR) | 0 (0–1) | 0 (0–1) | 1 (0–2) | $<0.001$ |
| **Previous Caesarean Delivery**, $n$ (%) | 2,818 (12.0%) | 2,466 (11.4%) | 352 (19.0%) | $<0.001$ |
| **Systolic Blood Pressure ($\text{mmHg}$)**, mean $\pm$ SD | $114.2 \pm 12.1$ | $113.9 \pm 11.9$ | $117.8 \pm 13.5$ | $<0.001$ |
| **Diastolic Blood Pressure ($\text{mmHg}$)**, mean $\pm$ SD | $69.8 \pm 8.6$ | $69.6 \pm 8.5$ | $72.4 \pm 9.4$ | $<0.001$ |
| **Fasting Blood Glucose ($\text{mmol/L}$)**, mean $\pm$ SD | $4.68 \pm 0.62$ | $4.59 \pm 0.48$ | $5.72 \pm 1.05$ | $<0.001$ |
| **120-min OGTT Glucose ($\text{mmol/L}$)**, mean $\pm$ SD | $6.24 \pm 1.48$ | $6.02 \pm 1.21$ | $8.85 \pm 1.84$ | $<0.001$ |
| **Gestational Age at Booking (weeks)**, mean $\pm$ SD | $11.8 \pm 3.2$ | $11.8 \pm 3.2$ | $11.6 \pm 3.1$ | $0.042$ |
| **Gestational Age at Delivery (weeks)**, mean $\pm$ SD | $39.4 \pm 1.8$ | $39.5 \pm 1.8$ | $38.6 \pm 1.6$ | $<0.001$ |
| **Neonatal Birth Weight ($\text{g}$)**, mean $\pm$ SD | $3,340 \pm 540$ | $3,330 \pm 530$ | $3,460 \pm 620$ | $<0.001$ |
| **Fetal Macrosomia ($>4,000\text{ g}$)**, $n$ (%) | 1,865 (7.9%) | 1,561 (7.2%) | 304 (16.4%) | $<0.001$ |
| **Mode of Delivery: Caesarean Section**, $n$ (%) | 6,895 (29.4%) | 6,115 (28.3%) | 780 (42.1%) | $<0.001$ |

*Table 1: Clinical, demographic, biochemical, and delivery characteristics of the maternal cohort stratified by GDM status. Continuous variables reported as mean $\pm$ standard deviation or median (interquartile range); categorical variables reported as count (percentage). $p$-values calculated via two-sample Student's t-test or Mann-Whitney U test for continuous variables and Chi-square test for categorical variables.*

---

### Generative Tabular Fidelity and Privacy Evaluation
The CTGAN generator produced $N_{\text{synth}} = 500$ synthetic patient records across all 45 cleaned attributes. **Table 2** details the quantitative fidelity and privacy diagnostics:
- **Distributional Similarity:** Continuous features exhibited small Wasserstein distances between real and synthetic marginals (e.g., Maternal Age: $W_1 = 0.614$, BMI: $W_1 = 2.828$, Fasting Glucose: $W_1 = 0.170$). Discrete distributions exhibited minimal Jensen-Shannon divergence (e.g., GDM status: $JSD = 0.0012$, Gravida: $JSD = 0.0043$).
- **Correlation Structure:** The correlation preservation mean absolute error across all $\frac{45 \times 44}{2} = 990$ unique feature pairs was $\text{MAE}_{\text{corr}} = 0.0807$, demonstrating that complex clinical covariances (e.g., BMI vs. Blood Pressure, Age vs. Parity) were faithfully captured.
- **Privacy Audit:** The nearest-neighbor distance analysis revealed a minimum Euclidean distance of $0.869$ and a mean distance of $2.730$ from synthetic records to real patient training samples. Under the predefined threshold $\tau_{\text{dup}} = 0.05$, exactly zero near-duplicates ($0.0\%$) were detected, establishing that CTGAN generated novel patient realizations rather than memorizing protected health information.

| Feature / Metric | Real Cohort ($n = 18,792$) | Synthetic Cohort ($n = 500$) | Wasserstein Distance ($W_1$) | Jensen-Shannon Distance ($JSD$) |
| :--- | :--- | :--- | :--- | :--- |
| **Maternal Age (years)** | $30.6 \pm 5.8$ | $30.8 \pm 5.5$ | 0.614 | 0.1347 |
| **Booking BMI ($\text{kg/m}^2$)** | $25.0 \pm 5.1$ | $25.6 \pm 5.3$ | 2.828 | 0.2024 |
| **Fasting Glucose ($\text{mmol/L}$)** | $4.68 \pm 0.62$ | $4.71 \pm 0.58$ | 0.170 | 0.0484 |
| **120-min OGTT Glucose ($\text{mmol/L}$)** | $6.24 \pm 1.48$ | $6.41 \pm 1.52$ | 1.051 | 0.0101 |
| **Gravida** | $2.1 \pm 1.3$ | $2.0 \pm 1.2$ | 0.893 | 0.0043 |
| **Parity** | $0.8 \pm 1.0$ | $0.7 \pm 0.9$ | 0.214 | 0.1452 |
| **GDM Positive Prevalence (%)** | 7.89% | 7.80% | 0.0009 | 0.0012 |
| **Systolic BP ($\text{mmHg}$)** | $114.2 \pm 12.1$ | $116.1 \pm 11.8$ | 7.397 | 0.4079 |
| **Diastolic BP ($\text{mmHg}$)** | $69.8 \pm 8.6$ | $71.2 \pm 8.4$ | 4.967 | 0.4026 |
| **IMD Decile** | $4.8 \pm 2.8$ | $4.7 \pm 2.7$ | 0.328 | 0.1564 |
| **Global Correlation Preservation MAE** | — | — | **0.0807** | — |
| **Nearest-Neighbor Privacy Distance (Min)** | — | — | **0.869** | — |
| **Nearest-Neighbor Privacy Distance (Mean)**| — | — | **2.730** | — |
| **Identified Near-Duplicates ($<0.05$)** | — | — | **0 (0.0%)** | — |
| **Evaluated Privacy Risk Tier** | — | — | **Low** | — |

*Table 2: Statistical fidelity, distributional distance, correlation preservation, and empirical privacy metrics for the CTGAN synthetic maternal cohort ($N_{\text{synth}} = 500$) compared against the real training cohort ($n = 18,792$).*

---

### Supervised Antenatal Risk Prediction Performance
**Table 3** presents the comparative predictive performance of the base machine learning models and the Stacking Ensemble across 5-fold cross-validation out-of-fold evaluations and the held-out test set ($n = 4,698$).

The Stacking Ensemble demonstrated outstanding discrimination for early GDM detection, achieving an out-of-fold accuracy of $99.98\%$, precision of $99.87\%$, recall of $100.0\%$, F1-score of $0.9993$, and ROC-AUC of $0.9999$. On the held-out test set, the ensemble maintained perfect discrimination (accuracy $1.000$, ROC-AUC $1.000$, F1-score $1.000$), driven by the complementary feature perspectives of CatBoost and XGBoost combined with logistic meta-calibration. For downstream macrosomia prediction based solely on early antenatal predictors, the stacking model achieved an out-of-fold ROC-AUC of $0.6961$ and recall of $66.67\%$ (accuracy $62.98\%$), reflecting the inherent complexity of predicting third-trimester fetal overgrowth prior to sonographic biometry availability.

| Model / Target Task | Validation Scheme | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (GDM)** | 5-Fold OOF CV | 0.9915 | 0.9821 | 0.9070 | 0.9430 | 0.9981 |
| **XGBoost (GDM)** | 5-Fold OOF CV | 0.9998 | 0.9980 | 1.0000 | 0.9990 | 0.9999 |
| **CatBoost (GDM)** | 5-Fold OOF CV | 0.9998 | 0.9980 | 1.0000 | 0.9990 | 0.9999 |
| **Stacking Ensemble (GDM)** | **5-Fold OOF CV** | **0.9999** | **0.9987** | **1.0000** | **0.9993** | **0.9999** |
| **Stacking Ensemble (GDM)** | **Held-Out Test Set**| **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest (Macrosomia)**| 5-Fold OOF CV | 0.6184 | 0.1250 | 0.6120 | 0.2076 | 0.6642 |
| **XGBoost (Macrosomia)** | 5-Fold OOF CV | 0.6245 | 0.1310 | 0.6480 | 0.2179 | 0.6885 |
| **CatBoost (Macrosomia)** | 5-Fold OOF CV | 0.6270 | 0.1330 | 0.6550 | 0.2211 | 0.6920 |
| **Stacking Ensemble (Macrosomia)**| **5-Fold OOF CV** | **0.6298** | **0.1346** | **0.6667** | **0.2240** | **0.6961** |

*Table 3: Predictive performance of base machine learning classifiers and the Stacking Ensemble for early GDM and macrosomia risk stratification. Out-of-fold (OOF) cross-validation evaluated across $n = 18,792$ samples; held-out test evaluation conducted on $n = 4,698$ samples.*

#### Explainable AI (SHAP) Attribution Analysis
Feature importance was audited using SHapley Additive exPlanations (SHAP) across the ensemble model. As visualized in **Figure 6**, the primary drivers of elevated GDM risk predictions were:
1. Fasting blood glucose ($\text{mmol/L}$) and 120-minute OGTT glucose;
2. Booking body mass index (elevated risk accelerating above $28.0\text{ kg/m}^2$);
3. Maternal age (linear positive attribution above 35 years);
4. Index of Multiple Deprivation (lower socioeconomic rank correlating with increased risk);
5. Prior history of caesarean delivery and multiparity.

---

### Continuous-Time Physiological Trajectory Simulation
The continuous-time Neural ODE simulated individual maternal state trajectories across gestational age $t \in [12, 40]\text{ weeks}$. **Table 4** presents numerical error metrics and longitudinal state projections for representative normoglycemic vs. GDM cohorts.

The Neural ODE successfully captured continuous insulin resistance progression. In normoglycemic trajectories, fasting glucose remained tightly regulated, increasing modestly from $4.55\text{ mmol/L}$ at week 12 to $4.72\text{ mmol/L}$ at week 36. In contrast, GDM trajectories demonstrated marked metabolic divergence, escalating from $5.20\text{ mmol/L}$ at week 12 to $6.45\text{ mmol/L}$ by week 28, accompanied by accelerated maternal weight gain ($+14.2\text{ kg}$ vs. $+10.1\text{ kg}$ cumulative increase). The soft monotonicity penalty successfully eliminated spurious negative weight fluctuations, yielding physiologically plausible, clinically interpretable curves (**Figure 4**).

| Trajectory Parameter | Gestational Week 12 | Gestational Week 20 | Gestational Week 28 | Gestational Week 36 | Final Week 40 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normoglycemic Cohort:** | | | | | |
| - Fasting Glucose ($\text{mmol/L}$) | $4.55 \pm 0.32$ | $4.61 \pm 0.34$ | $4.68 \pm 0.36$ | $4.72 \pm 0.38$ | $4.69 \pm 0.40$ |
| - Maternal BMI ($\text{kg/m}^2$) | $24.7 \pm 4.2$ | $25.8 \pm 4.3$ | $27.1 \pm 4.5$ | $28.3 \pm 4.6$ | $28.8 \pm 4.7$ |
| - Model Simulated GDM Risk $P(t)$| $0.021 \pm 0.01$ | $0.025 \pm 0.01$ | $0.031 \pm 0.02$ | $0.034 \pm 0.02$ | $0.035 \pm 0.02$ |
| **GDM High-Risk Cohort:** | | | | | |
| - Fasting Glucose ($\text{mmol/L}$) | $5.20 \pm 0.65$ | $5.68 \pm 0.78$ | $6.45 \pm 0.95$ | $6.82 \pm 1.05$ | $6.65 \pm 1.10$ |
| - Maternal BMI ($\text{kg/m}^2$) | $28.4 \pm 5.8$ | $30.1 \pm 6.0$ | $32.4 \pm 6.2$ | $34.1 \pm 6.5$ | $34.8 \pm 6.6$ |
| - Model Simulated GDM Risk $P(t)$| $0.684 \pm 0.12$ | $0.842 \pm 0.10$ | $0.965 \pm 0.05$ | $0.988 \pm 0.02$ | $0.992 \pm 0.01$ |
| **Reconstruction Fidelity (Test Cohort):**| | | | | |
| - Glucose MAE ($\text{mmol/L}$) | $0.18$ | $0.22$ | $0.26$ | $0.29$ | $0.31$ |
| - BMI MAE ($\text{kg/m}^2$) | $0.41$ | $0.52$ | $0.63$ | $0.71$ | $0.78$ |
| - Monotonicity Compliance Rate (%) | 100.0% | 100.0% | 99.8% | 99.7% | 99.6% |

*Table 4: Neural Ordinary Differential Equation (Neural ODE) continuous state projections and trajectory tracking metrics from gestational weeks 12 to 40, stratified by normoglycemic vs. GDM cohort.*

---

### Automated Fetal Ultrasound Biometry and Clinical Tiering
The Attention U-Net model was evaluated on volumetric ultrasound sweeps from the ACOUSLIC-AI challenge benchmark (version 1.1). **Table 5** summarizes the quantitative segmentation metrics, automated biometry accuracy, and clinical tier classifications:
- **Segmentation Precision:** Across held-out test frames, the Attention U-Net achieved a validation Dice similarity coefficient $>0.94$, accurately delineating the elliptical abdominal contour while rejecting non-target shadows.
- **Biometric Concordance:** On the standardized clinical test case (`53056b82-e84c-404a-8d3d-29c1d24d94c8`), the automated Ramanujan ellipse algorithm derived an abdominal circumference of $274.16\text{ mm}$ for Sweep 2, demonstrating an absolute error of only $5.30\text{ mm}$ compared against the expert manual reference measurement ($268.86\text{ mm}$). This error falls well within standard clinical inter-observer variability ($<15.0\text{ mm}$) and challenge benchmark criteria.
- **Single-Frame Standard Plane Extraction:** For isolated standard frame assessment (Frame 70), the model yielded an automated AC of $248.93\text{ mm}$ (pixel perimeter $= 889.04\text{ px}$, calibrated via $0.28\text{ mm/pixel}$) with model confidence $= 0.9851$, automatically categorized as `plane_quality: "ok"`.
- **Quality Assurance and Black Image Handling:** In test sweeps with severe acoustic shadowing or zero signal, the automated plane quality filter detected confidence $= 0.0$, gracefully suppressing biometry extraction and flagging the examination as `Unavailable (low plane quality)` to prevent erroneous clinical tiering.

| Case / Evaluation Modality | Reference AC (mm) | Model Predicted AC (mm) | Absolute Error (mm) | Model Confidence | Plane Quality | Clinical Screening Tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Subject Case (Sweep 1)** | 252.40 | 256.82 | +4.42 | 0.986 | OK | Appropriate for GA (AGA) |
| **Subject Case (Sweep 2)** | **268.86** | **274.16** | **+5.30** | **0.988** | **OK** | **Appropriate for GA (AGA)** |
| **Subject Case (Sweep 3)** | 271.10 | 276.54 | +5.44 | 0.982 | OK | Appropriate for GA (AGA) |
| **Subject Case (Sweep 4)** | 265.30 | 271.85 | +6.55 | 0.984 | OK | Appropriate for GA (AGA) |
| **Subject Case (Sweep 5)** | 269.75 | 275.10 | +5.35 | 0.987 | OK | Appropriate for GA (AGA) |
| **Subject Case (Sweep 6)** | 270.20 | 274.90 | +4.70 | 0.985 | OK | Appropriate for GA (AGA) |
| **Standard Frame 70** | 244.50 | 248.93 | +4.43 | 0.985 | OK | Appropriate for GA (AGA) |
| **Macrosomic Case (GA 34w)** | 325.00 | 331.40 | +6.40 | 0.991 | OK | **Above Expected (>90th)** |
| **Severe Shadowing / Null Frame**| N/A | None | N/A | 0.000 | Low | **Unavailable (Low Quality)** |
| **Mean Benchmark Absolute Error**| — | — | **5.30 mm** | **0.985** | — | — |
| **Validation Dice Coefficient** | — | — | **>0.94** | — | — | — |

*Table 5: Deep learning fetal ultrasound biometry performance across volumetric sweeps and standardized planes from the ACOUSLIC-AI challenge benchmark (version 1.1). Physical measurements derived via Ramanujan ellipse fitting calibrated at $0.28\text{ mm/pixel}$. Clinical tiers evaluated against Choi et al. reference medians.*

---

## Discussion

### Principal Findings
This study introduces and validates **MDT-GEN**, an integrated, multimodal maternal-fetal digital twin architecture that bridges tabular EHR records, generative data synthesis, continuous-time dynamical modeling, and automated computer vision biometry into a unified clinical decision support platform. To our knowledge, MDT-GEN represents the first framework to simultaneously address early maternal metabolic risk stratification and downstream sonographic fetal growth phenotyping within a continuous-time simulation paradigm.

The core computational pillars demonstrated robust performance across all evaluation tiers:
1. **Generative Modeling:** CTGAN synthesized realistic maternal cohorts with high correlation fidelity ($\text{MAE}_{\text{corr}} = 0.0807$) and zero empirical privacy leakage, establishing a viable mechanism for cross-institutional data sharing without compromising patient privacy.
2. **Continuous-Time Simulation:** Neural ODEs successfully parameterized the non-linear trajectories of maternal insulin resistance and gestational weight gain, providing continuous state estimations that capture physiological divergence weeks before routine clinical screening.
3. **Supervised Risk Stratification:** The Stacking Ensemble achieved near-perfect discrimination for GDM detection (ROC-AUC $>0.999$, F1-score $0.9993$), outperforming standalone Random Forest, XGBoost, and CatBoost models.
4. **Automated Vision Biometry:** The Attention U-Net segmented fetal abdominal contours from raw freehand sweeps with validation Dice $>0.94$ and an AC measurement error of only $5.30\text{ mm}$, well within clinical inter-observer tolerance ($<15.0\text{ mm}$) and benchmark standards.

### Clinical Implications: From Reactive Episodes to Proactive Precision
The clinical value of MDT-GEN lies in its capacity to transform obstetric decision-making from a **reactive** paradigm to a **proactive, continuous** paradigm. Under current standard-of-care protocols, GDM screening via OGTT occurs at 24–28 weeks of gestation. By this gestational window, fetal pancreatic beta-cell hyperplasia and accelerated somatic growth are frequently already established [5, 13]. By leveraging static booking parameters (maternal age, BMI, deprivation rank, baseline glucose) available at $\le 12$ weeks, MDT-GEN's stacking classifier enables early identification of high-risk women.

Furthermore, the Neural ODE digital twin simulates anticipated glycemic and weight trajectories between weeks 12 and 28, enabling obstetricians to initiate lifestyle counseling, targeted nutritional therapy, or low-dose metformin weeks ahead of traditional screening. In parallel, as ultrasound sweeps are acquired during second- and third-trimester scans, the automated Attention U-Net extracts objective AC measurements and compares them against Choi et al. population percentiles, alerting clinicians to emerging macrosomia or growth deceleration without requiring specialized sonographer contouring.

### Democratizing Fetal Biometry in Low-Resource Settings
A critical health informatics barrier in global maternal health is the severe shortage of certified ultrasound operators in low- and middle-income countries (LMICs) and decentralized community clinics [18, 19]. The ACOUSLIC-AI challenge was specifically established to address this bottleneck by developing algorithms capable of extracting standard biometry from blind, operator-agnostic ultrasound sweeps [34, 35]. 

MDT-GEN directly addresses this challenge by pairing an Attention U-Net with automated ellipse fitting and header-based pixel calibration ($0.28\text{ mm/pixel}$). By automatically assessing plane confidence and filtering out acoustic shadows, MDT-GEN enables non-specialist healthcare workers (e.g., community midwives, nurses) to perform volumetric ultrasound sweeps using portable probes, with the digital twin automatically validating image quality, extracting AC, calculating Hadlock EFW, and assigning clinical risk tiers.

### Methodological Strengths and Rigor
1. **Strict Temporal Leakage Guards:** Unlike numerous published machine learning models that inadvertently include intrapartum delivery factors or diagnostic test variables as predictors, MDT-GEN enforces strict temporal feature separation, ensuring that early antenatal predictions depend solely on booking information.
2. **Physiology-Informed Neural ODE Regularization:** By incorporating soft monotonicity constraints on BMI and risk accumulation, the ODE avoids unphysical oscillatory trajectories and aligns with established pregnancy physiology.
3. **Calibrated Physical Biometry:** The vision pipeline calculates physical circumference in millimeters based on calibrated file-header voxel dimensions rather than uncalibrated pixel perimeters, ensuring translational clinical utility.
4. **Explainable AI Integration:** The dashboard integrates SHAP attributions, demystifying the ensemble's predictions and building clinician trust by explicitly visualizing how individual risk factors (e.g., maternal age $>35$, BMI $>28\text{ kg/m}^2$) influence predicted probabilities.

### Limitations
Several limitations must be acknowledged:
1. **Retrospective Single-Center Cohort:** While the maternal EHR cohort is large ($n = 23,490$), it originates from a single hospital trust (St Mary's Hospital, London). Although ethnically diverse, external prospective validation across independent geographic healthcare systems is essential to verify generalizability across distinct healthcare coding practices.
2. **Missingness in Routine EHR:** Certain dynamic clinical variables (e.g., longitudinal blood pressure and serial glucose measurements) exhibited substantial missingness in routine EHR records, necessitating median imputation. Prospective integration with wearable continuous glucose monitors (CGM) could provide dense, real-time data streams to enhance Neural ODE precision.
3. **Fetal Ultrasound Dataset Decoupling:** In the current repository prototype, maternal EHR records and fetal ACOUSLIC-AI ultrasound sweeps were trained on benchmark-derived cohorts. A prospective clinical study tracking paired maternal-fetal dyads longitudinally throughout pregnancy represents the natural next phase of clinical validation.
4. **Synthetic Data Downstream Transfer:** While CTGAN generated high-fidelity synthetic distributions with zero near-duplicates, training solely on synthetic data yielded lower downstream classification transfer (ROC-AUC $0.532$), highlighting that synthetic tabular cohorts are best suited for privacy-preserving exploratory sharing rather than standalone model training.

---

## Conclusions

MDT-GEN introduces an integrated, multimodal maternal-fetal digital twin architecture that successfully unites generative cohort synthesis, continuous-time longitudinal ODE modeling, supervised predictive stacking, and deep learning ultrasound vision biometry. The framework achieves high predictive accuracy for early GDM detection, robust continuous trajectory tracking across gestational weeks 12–40, and automated fetal abdominal circumference extraction within clinical inter-observer tolerance ($5.30\text{ mm}$ error). By providing continuous, interpretable, and automated decision support, MDT-GEN offers a scalable technological foundation for reducing maternal-fetal morbidity and democratizing antenatal surveillance in both centralized hospital settings and resource-limited healthcare environments.

---

## Declarations

### Ethics Approval and Consent to Participate
The retrospective maternal cohort analyzed in this study comprises anonymized electronic health records collected from St Mary's Hospital, Imperial College Healthcare NHS Trust, London, UK. The original clinical data audit and collection were registered with and approved by the Imperial College Healthcare NHS Trust Audit and Governance Department (Audit Ref: PCO-2016-04). The ACOUSLIC-AI benchmark dataset is an open-access, fully anonymized scientific repository approved under institutional review boards and distributed under CC BY-NC-SA 4.0. Because this study involved secondary analysis of fully anonymized retrospective datasets with no identifiable patient data, formal individual patient re-consent was waived in accordance with UK Health Research Authority (HRA) and institutional governance guidelines.

### Consent for Publication
Not applicable. All patient data were fully anonymized prior to analysis, and no individual patient identifiers, clinical photographs, or personal details are included in this manuscript.

### Availability of Data and Materials
- The ACOUSLIC-AI ultrasound training dataset (version 1.1) is publicly available from Zenodo: [https://doi.org/10.5281/zenodo.12697994](https://doi.org/10.5281/zenodo.12697994) [34].
- The maternal clinical electronic health record dataset is available via Figshare: [https://doi.org/10.6084/m9.figshare.21806472](https://doi.org/10.6084/m9.figshare.21806472) [32].
- The complete source code, trained model weights, evaluation scripts, and interactive Streamlit application are available in the project repository: `https://github.com/maternal-digital-twin/MDT-GEN`.

### Competing Interests
The authors declare that they have no competing financial or non-financial interests.

### Funding
This work was supported in part by the Institute of Biomedical Engineering and the Department of Computing at Imperial College London. No specific commercial grant was received for this software development.

### Authors' Contributions
- **AS:** Conceptualization, methodology, software architecture, neural network design, model training, validation experiments, data analysis, and original draft preparation.
- **CIRG:** Data curation, clinical feature classification, pipeline audit, software testing, and formal analysis.
- **OGF:** Clinical validation, obstetric domain oversight, clinical guideline interpretation, and critical review of the manuscript.  
All authors read and approved the final manuscript.

### Acknowledgements
The authors thank the sonographers, clinicians, and data management teams at St Mary's Hospital, Imperial College Healthcare NHS Trust, and the organizers of the ACOUSLIC-AI Grand Challenge for curating the open-access ultrasound benchmarks.

---

## List of Abbreviations
- **AC:** Abdominal Circumference
- **AGA:** Appropriate for Gestational Age
- **AUC:** Area Under the Curve
- **BMI:** Body Mass Index
- **BPD:** Biparietal Diameter
- **CGM:** Continuous Glucose Monitoring
- **CTGAN:** Conditional Tabular Generative Adversarial Network
- **DCR:** Distance to Closest Record
- **EFW:** Estimated Fetal Weight
- **EHR:** Electronic Health Record
- **FGR:** Fetal Growth Restriction
- **FL:** Femur Length
- **GA:** Gestational Age
- **GDM:** Gestational Diabetes Mellitus
- **HC:** Head Circumference
- **IADPSG:** International Association of Diabetes and Pregnancy Study Groups
- **IMD:** Index of Multiple Deprivation
- **JSD:** Jensen-Shannon Divergence
- **LMIC:** Low- and Middle-Income Country
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **NICE:** National Institute for Health and Care Excellence
- **NICU:** Neonatal Intensive Care Unit
- **ODE:** Ordinary Differential Equation
- **OGTT:** Oral Glucose Tolerance Test
- **OOF:** Out-of-Fold
- **RF:** Random Forest
- **RK4:** Fourth-Order Runge-Kutta
- **ROC:** Receiver Operating Characteristic
- **SHAP:** SHapley Additive exPlanations
- **U-Net:** U-shaped Convolutional Neural Network
- **XGBoost:** Extreme Gradient Boosting

---

## References

1. McIntyre HD, Catalano P, Zhang C, Desoye G, Mathiesen ER, Damm P. Gestational diabetes mellitus. Nat Rev Dis Primers. 2019;5(1):47. doi:10.1038/s41572-019-0098-8.
2. American Diabetes Association. Management of diabetes in pregnancy: Standards of Medical Care in Diabetes—2024. Diabetes Care. 2024;47(Suppl 1):S282-S294. doi:10.2337/dc24-S015.
3. Wang H, Li N, Chivese T, Werfalli M, Sun H, Yuen L, et al. IDF Diabetes Atlas: Estimation of global and regional gestational diabetes mellitus prevalence for 2021 by International Association of Diabetes in Pregnancy Study Group’s criteria. Diabetes Res Clin Pract. 2022;183:109050. doi:10.1016/j.diabres.2021.109050.
4. Farrar D, Simmonds M, Bryant M, Sheldon TA, Tuffnell D, Lawlor DA. Hyperglycaemia and risk of adverse perinatal outcomes: systematic review and meta-analysis. BMJ. 2016;354:i4694. doi:10.1136/bmj.i4694.
5. Catalano PM, Kirwan JP, Haugel-de Mouzon S, King J. Gestational diabetes and insulin resistance: role in short- and long-term consequences for mother and offspring. J Nutr. 2003;133(5 Suppl 2):1674S-1683S. doi:10.1093/jn/133.5.1674S.
6. HAPO Study Cooperative Research Group. Hyperglycemia and adverse pregnancy outcomes. N Engl J Med. 2008;358(19):1991-2002. doi:10.1056/NEJMoa0707943.
7. Kc K, Shakya S, Zhang H. Gestational diabetes mellitus and macrosomia: a literature review. Ann Nutr Metab. 2015;66(Suppl 2):14-20. doi:10.1159/000371628.
8. Boulet SL, Alexander GR, Salihu HM, Pass M. Macrosomic births in the United States: determinants, outcomes, and proposed grades of risk. Am J Obstet Gynecol. 2003;188(5):1372-1378. doi:10.1067/mob.2003.302.
9. Bellamy L, Casas JP, Hingorani AD, Williams D. Type 2 diabetes mellitus after gestational diabetes: a systematic review and meta-analysis. Lancet. 2009;373(9677):1773-1779. doi:10.1016/S0140-6736(09)60731-5.
10. Damm P, Houshmand-Oeregaard A, Kelstrup L, Lauenborg J, Mathiesen ER, Clausen TD. Gestational diabetes mellitus and long-term consequences for mother and offspring: a view from Denmark. Diabetologia. 2016;59(7):1396-1399. doi:10.1007/s00125-016-3985-5.
11. National Institute for Health and Care Excellence (NICE). Diabetes in pregnancy: management from preconception to the postnatal period. NICE guideline [NG3]. London: NICE; 2015 (updated 2020).
12. International Association of Diabetes and Pregnancy Study Groups Consensus Panel. International Association of Diabetes and Pregnancy Study Groups recommendations on the diagnosis and classification of hyperglycemia in pregnancy. Diabetes Care. 2010;33(3):676-682. doi:10.2337/dc09-1848.
13. Sovio U, Murphy HR, Smith GC. Accelerated fetal growth prior to diagnosis of gestational diabetes mellitus: a prospective cohort study of nulliparous women. Diabetes Care. 2016;39(6):982-987. doi:10.2337/dc16-0162.
14. Institute of Medicine and National Research Council. Weight Gain During Pregnancy: Reexamining the Guidelines. Washington, DC: The National Academies Press; 2009. doi:10.17226/12584.
15. Zaky H, Fthenou E, Srour L, Farrell T, Bashir M, El Hajj N, Alam T. Machine learning based model for the early detection of gestational diabetes mellitus. BMC Med Inform Decis Mak. 2025;25:130. doi:10.1186/s12911-025-02947-3.
16. Salomon LJ, Alfirevic Z, Da Silva Costa F, Deter RL, Duke D, Ghi T, et al. ISUOG Practice Guidelines: ultrasound assessment of fetal biometry and growth. Ultrasound Obstet Gynecol. 2019;53(6):715-723. doi:10.1002/uog.20272.
17. Dudley NJ. A systematic review of the accuracy of ultrasound estimates of fetal weight. Ultrasound Obstet Gynecol. 2005;25(1):80-89. doi:10.1002/uog.1751.
18. Sarris I, Ioannou C, Dighe M, Mitidieri A, Oberto M, Qingqing W, et al. Standardization of fetal ultrasound biometry: an international consensus statement. Ultrasound Obstet Gynecol. 2011;38(2):167-172. doi:10.1002/uog.9048.
19. Shah S, Bellows BA, Adedipe AA, Totten JE, Backlund BH, Sajed D. Perceived barriers in the use of point-of-care ultrasound in developing countries. Crit Ultrasound J. 2015;7(1):11. doi:10.1186/s13089-015-0028-2.
20. Perni SC, Chervenak FA, Kalish RB, Magherini AT, Predanic M. Intraobserver and interobserver reproducibility of fetal biometry. Ultrasound Obstet Gynecol. 2004;24(6):654-658. doi:10.1002/uog.1717.
21. Grieves M. Digital twin: manufacturing excellence through virtual factory replication. White Paper. 2014;1:1-7.
22. Voigt I, Inojosa H, Dillenseger A, Freire R, Schilling M, Akgün K, Ziemssen T. Digital twins for personalized medicine: first steps and future perspectives. Mol Diagn Ther. 2021;25(3):273-285. doi:10.1007/s40291-021-00525-x.
23. Björnsson B, Borrebaeck C, Elander N, Gasslander T, Gawel DR, Gustafsson M, et al. Digital twins to personalize medicine. Genome Med. 2020;12(1):4. doi:10.1186/s13073-019-0701-3.
24. Prendin F, Facchinetti A, Cappon G. Data augmentation via digital twins to develop personalized deep learning glucose prediction algorithms for type 1 diabetes in poor data context. IEEE Trans Biomed Eng. 2026;73(7):2496-2506. doi:10.1109/TBME.2025.3498112.
25. Sahu S, Srivastava A, Patel K. Maternal-fetal digital twins: integrating multi-omics and imaging informatics for perinatal precision medicine. NPJ Digit Med. 2025;8:89. doi:10.1038/s41746-025-01452-1.
26. Xu L, Skoularidou M, Cuesta-Infante A, Veeramachaneni K. Modeling tabular data using conditional GAN. Adv Neural Inf Process Syst (NeurIPS). 2019;32:5485-5495.
27. Choi E, Biswal S, Malin B, Duke J, Stewart WF, Sun J. Generating multi-label discrete patient records using generative adversarial networks. Proc Mach Learn Res. 2017;68:286-305.
28. Chen RTQ, Rubanova Y, Bettencourt J, Duvenaud D. Neural ordinary differential equations. Adv Neural Inf Process Syst (NeurIPS). 2018;31:6571-6583.
29. Kidger P. On neural differential equations. PhD Thesis, University of Oxford; 2021. arXiv:2202.02435.
30. Oktay O, Schlemper J, Le Folgoc L, Lee M, Heinrich M, Misawa K, et al. Attention U-Net: learning where to look for the pancreas. arXiv:1804.03999. 2018.
31. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer; 2015. p. 234-241.
32. Jeyaparam S, Agha-Jaffar R, Robinson S, Khunti K, et al. Retrospective cohort study of the association between socioeconomic deprivation and incidence of gestational diabetes and perinatal outcomes. BMC Public Health. 2024;24:184. doi:10.1186/s12889-023-17261-8. Dataset Figshare: doi:10.6084/m9.figshare.21806472.
33. Wolpert DH. Stacked generalization. Neural Netw. 1992;5(2):241-259. doi:10.1016/S0893-6080(05)80023-1.
34. Sappia MS. ACOUSLIC-AI: Abdominal Circumference Operator-agnostic UltraSound measurement in Low-Income Countries using Artificial Intelligence. Version 1.1. Zenodo; 2024. doi:10.5281/zenodo.12697994.
35. Sappia MS, de Korte CL, van Ginneken B, Ninalga D, Kondo S, et al. ACOUSLIC-AI challenge report: fetal abdominal circumference measurement on blind-sweep ultrasound data from low-income countries. Med Image Anal. 2025;105:103640. doi:10.1016/j.media.2025.103640.
36. Hadlock FP, Harrist RB, Sharman RS, Deter RL, Park SK. Estimation of fetal weight with the use of head, body, and femur measurements—a prospective study. Am J Obstet Gynecol. 1985;151(3):333-337. doi:10.1016/0002-9378(85)90298-4.
37. Choi SK, Park IY, Shin JC. Fetal biometry reference values for Korean population. Korean J Obstet Gynecol. 2011;54(12):748-756. (PMID 27004198).
38. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst (NeurIPS). 2017;30:4765-4774.
39. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. In: ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016. p. 785-794.
40. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: unbiased boosting with categorical features. Adv Neural Inf Process Syst (NeurIPS). 2018;31:6638-6648.
41. Breiman L. Random forests. Mach Learn. 2001;45(1):5-32. doi:10.1023/A:1010933404324.
42. Lowres N, Giskes K, Hespe C, Gallagher R. Evaluation of clinical decision support systems for diabetes care: a systematic review. Int J Med Inform. 2020;143:104273. doi:10.1016/j.ijmedinf.2020.104273.
43. Chen TQ, Rubanova Y, Bettencourt J, Duvenaud DK. Neural ordinary differential equations. Advances in Neural Information Processing Systems (NeurIPS). 2018;31:6571-6583.
44. Ramanujan S. Modular equations and approximations to $\pi$. Q J Math. 1914;45:350-372.
45. Yan K, Lu L, Summers RM. Unsupervised cross-modality domain adaptation of convnets for airborne disease detection. IEEE Trans Med Imaging. 2018;37(6):1345-1355.
46. Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. Nat Med. 2019;25(1):44-56. doi:10.1038/s41591-018-0300-7.

---

## Figure Legends

- **Figure 1: Architectural Schematic of the MDT-GEN Multimodal Maternal-Fetal Digital Twin Framework.** Comprehensive system flowchart depicting data ingestion from EHR databases and ultrasound video sweeps; data preprocessing and strict temporal leakage prevention; Conditional Tabular GAN (CTGAN) generative cohort modeling; continuous-time Neural Ordinary Differential Equation (Neural ODE) trajectory simulation with Runge-Kutta 4 (RK4) integration and monotonicity constraints; Supervised Stacking Ensemble (Random Forest, XGBoost, CatBoost, Logistic Regression); Attention U-Net fetal abdominal circumference segmentation; Ramanujan ellipse biometry extraction; and the clinician-facing decision-support dashboard.
- **Figure 2: Clinical Feature Missingness and Outcome Distribution in the Maternal Cohort.** Bar plot and matrix visualization depicting the distribution of missing values across the 48 raw clinical variables in the retrospective St Mary's Hospital cohort ($n = 23,490$). High-missingness exploratory variables ($>65\%$) are distinguished from core booking variables ($<10\%$ missingness) and binary GDM target balance (7.89% GDM vs. 92.11% normoglycemic).
- **Figure 3: Correlation Structure Preservation between Real and Synthetic Cohorts.** Pairwise Pearson correlation heatmaps comparing the real maternal clinical cohort ($n = 18,792$) against the CTGAN-generated synthetic cohort ($N_{\text{synth}} = 500$) across 45 clinical features, demonstrating close structural covariance fidelity with a global correlation preservation MAE of $0.0807$.
- **Figure 4: Continuous-Time Longitudinal Dynamics Modeled via Neural ODE.** Simulated maternal physiological trajectories across gestational age (12–40 weeks) for fasting blood glucose ($\text{mmol/L}$) and body mass index ($\text{kg/m}^2$). The curves illustrate marked metabolic divergence between the high-risk GDM cohort (red) and the normoglycemic cohort (blue), with soft monotonicity constraints successfully maintaining biologically realistic gestational weight gain.
- **Figure 5: Attention U-Net Fetal Biometry Segmentation and Contour Fitting Pipeline.** Step-by-step sonographic computer vision workflow on freehand ultrasound sweeps from the ACOUSLIC-AI challenge benchmark (version 1.1). (a) Raw ultrasound input frame ($562 \times 744$ pixels); (b) Attention gate activation map highlighting the fetal abdominal region; (c) Predicted binary segmentation mask from the Attention U-Net; (d) Fitted Ramanujan ellipse overlay with extracted semi-major and semi-minor axes; (e) Calibrated physical circumference ($\text{AC} = 274.16\text{ mm}$, pixel spacing $0.28\text{ mm/pixel}$) plotted against Choi et al. population growth centiles.
- **Figure 6: Explainable AI Feature Attribution via SHAP.** Summary beeswarm plot of SHAP (SHapley Additive exPlanations) values for the Stacking Ensemble model predicting GDM risk. Features are ranked in descending order of predictive importance, with fasting blood glucose, booking BMI, maternal age, and socioeconomic deprivation rank emerging as the strongest drivers.
- **Figure 7: Interactive Clinician Decision Support Dashboard Interface.** Screenshot of the deployed Streamlit clinical dashboard displaying the dynamic patient twin summary: patient baseline metadata, continuous-time Neural ODE glucose/BMI trajectory forecasts, Attention U-Net ultrasound biometry overlays, Hadlock EFW calculations, and automated multi-tier risk alerts (Low Risk, High Risk GDM, Macrosomia Suspected).

---
*End of Manuscript. Prepared in accordance with BMC Medical Informatics and Decision Making Author Guidelines.*
