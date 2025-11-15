Here is the README content you provided, formatted in Markdown.

***

# 🧠 On The Role of K-Space Acquisition in MRI Reconstruction Domain Generalization

**Authors:**
Mohammed Wattad, Tamir Shor, Alex Bronstein
*Faculty of Computer Science, Technion – Israel Institute of Technology*
📧 `mohammed-wa@campus.technion.ac.il`, `tamir.shor@campus.technion.ac.il`, `bron@cs.technion.ac.il`

---

## 📄 Overview

This repository accompanies the paper:

**"On The Role of K-Space Acquisition in MRI Reconstruction Domain-Generalization"**
*AAAI 2026 (W3PHIAI Workshop Submission)*

Our work investigates how learned k-space sampling trajectories influence the domain generalization ability of MRI reconstruction models.
We show that acquisition-aware regularization—through stochastic or adversarial perturbations of k-space trajectories—can significantly improve reconstruction robustness under cross-domain shifts.

---

## 🚀 Key Contributions

### Systematic Cross-Domain Evaluation
We compare reconstruction models trained with fixed vs learned k-space sampling patterns across multiple domains.

### Trajectory-Aware Domain Generalization (DG)
We propose and analyze three acquisition-level regularization strategies:

* **Random Trajectory Noise** – simulates natural acquisition imperfections.
* **Adversarial Trajectory Noise** – enforces robustness against worst-case trajectory deviations.
* **Image Noise Baseline** – standard image-domain augmentation for comparison.

### Learned Sampling Improves Robustness
Learned k-space trajectories exhibit significantly improved generalization across unseen MRI domains—even when not used at inference time.

### Novel Perspective
We treat the acquisition process itself as a tunable component for domain robustness, not just for acceleration.
