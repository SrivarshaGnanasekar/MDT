# MDT-GEN Formula References

This document distinguishes published equations from project-defined model choices. It describes the current implementation and does not establish clinical validation.

## 1. Fetal Segmentation and AC

Ramanujan's approximation for an ellipse with semi-axes $a$ and $b$ is:

$$C_{px} = \pi\left[3(a+b)-\sqrt{(3a+b)(a+3b)}\right]$$

When calibrated pixel spacing is available, physical length may be derived from pixel measurements. For isotropic spacing $s$ in cm/pixel, $C_{cm}=C_{px}s$. Non-square pixels require separate horizontal and vertical calibration.

**Current implementation limitation:** the U-Net predicts background, artery, vein, liver, and stomach. It does not segment the abdominal contour. The extractor's fitted organ-envelope perimeter is therefore not fetal AC; it is stored in pixels only. The data do not provide pixel calibration, so physical organ areas/perimeters and AC are left unavailable rather than estimated using an arbitrary scale.

## 2. Gestational-Age AC Reference

The project uses the regression reference attributed to Choi et al.:

$$AC_{ref}(cm)=1.013\,GA(weeks)-4.634$$

This is a reference median, not a distribution. If a valid measured AC is supplied, analytical deviations are:

$$\Delta AC=AC_{measured}-AC_{ref}$$

$$\Delta AC_{relative}(\%)=\frac{AC_{measured}-AC_{ref}}{AC_{ref}}\times100$$

These are supporting features, not diagnoses or percentiles. A percentile or z-score needs a gestational-age-specific reference distribution (for example, its mean and standard deviation); the median equation alone cannot provide one. The current U-Net output is not a valid measured AC.

## 3. Hadlock EFW

When all measurements are available in centimeters, the implemented Hadlock estimate is:

$$\log_{10}(EFW_g)=1.3596+0.0064(HC)+0.0424(AC)+0.174(FL)+0.00061(BPD)(AC)-0.00386(AC)(FL)$$

The implementation reports $EFW_g=10^{\log_{10}(EFW_g)}$ only when HC, AC, FL, and BPD are all supplied. The Attention U-Net does not produce these four measurements; otherwise EFW is unavailable.

## 4. Stacking Classifier and Decision Threshold

For the RF, XGBoost, and CatBoost base learners, the meta-feature vector is:

$$\mathbf{z}_i=[P^{OOF}_{RF,i},P^{OOF}_{XGB,i},P^{OOF}_{Cat,i}]^T$$

Out-of-fold (OOF) probabilities train the logistic-regression meta-learner:

$$P(y=1\mid\mathbf{z})=\sigma(\mathbf{w}^T\mathbf{z}+b),\qquad \sigma(x)=\frac{1}{1+e^{-x}}$$

The class decision is $\hat y=\mathbb{1}[P\geq\tau]$. The current default is $\tau=0.50$; it is a model operating threshold, not a clinically validated threshold. AC deviation does not override the ensemble probability.

## 5. Attention Gate

The attention gate follows the core additive formulation of Oktay et al.:

$$q_{attn}=\psi^T\left[\sigma_1(W_x^Tx+W_g^Tg+b_g)\right]+b_\psi$$

$$\alpha=\sigma_2(q_{attn}),\qquad \hat{x}=\alpha\odot x$$

Here $x$ is the encoder skip feature, $g$ is the decoder gating signal, and $\sigma_2$ is sigmoid. The implementation uses ReLU for $\sigma_1$, learned $1\times1$ convolutions for the projections, and a sigmoid coefficient. Normalization in the surrounding U-Net convolution blocks is implementation-specific; no claim is made that the whole network omits Batch Normalization.

## 6. Segmentation Loss

For $N$ pixels and $C$ classes, cross-entropy is:

$$L_{CE}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}y_{ic}\log(p_{ic}+\epsilon)$$

The class-wise soft Dice score and loss average over the foreground classes in the current training configuration:

$$Dice_c=\frac{2\sum_i p_{ic}y_{ic}+\epsilon}{\sum_i p_{ic}+\sum_i y_{ic}+\epsilon},\qquad Dice=\frac{1}{C-1}\sum_{c=1}^{C-1}Dice_c$$

$$L_{Dice}=1-Dice,\qquad L_{seg}=\lambda_{CE}L_{CE}+\lambda_{Dice}L_{Dice}$$

The current training defaults set both weights to $1.0$. These are implementation hyperparameters, not medical constants.

## 7. Neural ODE

The continuous-time model is:

$$\frac{d\mathbf{y}(t)}{dt}=f_\theta(\mathbf{y}(t),t)$$

The current state is $\mathbf{y}(t)=[Glucose(t),BMI(t)]$. Learned risk is:

$$P_{GDM}(t)=\sigma(\mathbf{w}_r^T\mathbf{y}(t)+b_r)$$

The implemented RK4 solver uses:

$$k_1=f_\theta(y_t,t),\quad k_2=f_\theta(y_t+\tfrac{h}{2}k_1,t+\tfrac{h}{2})$$
$$k_3=f_\theta(y_t+\tfrac{h}{2}k_2,t+\tfrac{h}{2}),\quad k_4=f_\theta(y_t+hk_3,t+h)$$
$$y_{t+h}=y_t+\frac{h}{6}(k_1+2k_2+2k_3+k_4)$$

The BMI regularizer is a soft non-decreasing trajectory constraint. In the current trainer the derivative is evaluated at each batch's mean booking week:

$$L_{BMI\_mono}=\lambda_{mono}\,\mathbb{E}_t\left[\max\left(0,-\frac{dBMI(t)}{dt}\right)^2\right]$$

It discourages downward trajectories; it is not a clinical law. The code also has a project-defined penalty discouraging predicted GDM risk from decreasing between booking and delivery.

The trajectory and binary-classification terms use MSE and binary cross-entropy:

$$MSE=\frac{1}{N}\sum_i(y_i-\hat y_i)^2$$

$$L_{GDM}=-\left[y\log P_{GDM}+(1-y)\log(1-P_{GDM})\right]$$

The implemented objective is:

$$L_{total}=\lambda_{Glu}L_{MSE,Glu}+\lambda_{BMI}L_{MSE,BMI}+\lambda_{GDM}L_{GDM}+\lambda_{mono}(L_{BMI\_mono}+L_{risk\_mono})$$

Its weights are training hyperparameters (defaults: $1.0$, $0.1$, $2.0$, and $0.5$, respectively), not clinical constants. MAE is not currently part of this training objective.

## References

- Choi et al., gestational-age fetal biometric reference equations, PMID 27004198.
- Hadlock et al., fetal-weight estimation, PMID 3881966.
- Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, arXiv:1804.03999.
- Chen et al., *Neural Ordinary Differential Equations*, arXiv:1806.07366.
