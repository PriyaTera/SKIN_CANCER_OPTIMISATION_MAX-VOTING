# SKIN_CANCER_OPTIMISATION_MAX-VOTING
1. Project Overview
The README opens with a concise one-paragraph project description emphasizing reproducibility and methodological transparency. It clearly states that the framework implements seven distinct XAI methods (Grad-CAM, Integrated Gradients, LIME, Saliency Maps, DeepLIFT, SmoothGrad, and SHAP) integrated with an ensemble classification pipeline (Random Forest, XGBoost, LightGBM) using max voting aggregation on the HAM10000 and ISIC 2018 datasets.
2. Installation & Dependencies
A dedicated section now provides explicit pip installation commands for all required packages (TensorFlow, scikit-learn, LIME, SHAP, mahotas, XGBoost, LightGBM, etc.), ensuring reproducibility across computational environments. Python version requirements (3.8+) are clearly specified.
3. Dataset Setup Instructions
Step-by-step guidance for downloading and organizing both HAM10000 and ISIC 2018 datasets, including direct links to official sources (Harvard Dataverse, ISIC Archive) and expected directory structures. This removes ambiguity about data placement and preprocessing.
4. Repository Structure
A hierarchical diagram showing the organization of modules:
•	data/: Dataset loading and preprocessing
•	models/: Individual model implementations (RF, XGBoost, LightGBM) and ensemble aggregation
•	xai_methods/: All seven XAI implementations (Grad-CAM, Integrated Gradients, LIME, Saliency Maps, DeepLIFT, SmoothGrad, SHAP)
•	evaluation/: Metrics computation and confusion matrix visualization
•	output/: Results and figures
Legacy/exploratory code has been explicitly isolated in an /archive/ directory with a clear note that it is not part of the reproducible pipeline.
5. Feature Extraction Pipeline
Detailed documentation of the hand-crafted feature extraction process, including:
•	Color histograms (RGB channels)
•	Color moments (mean, standard deviation, skewness)
•	GLCM texture features (contrast, correlation, energy, homogeneity)
•	Local Binary Pattern (LBP) descriptors
•	Total feature dimensionality and preprocessing steps (resizing to 256×256, RGB normalization, Gaussian blurring)
6. Genetic Algorithm for Feature Selection
Clear explanation of the GA optimization procedure used to select informative features from the 800+ extracted features, including population size (15), generations (8), crossover probability (0.8), and mutation probability (0.1). This ensures transparency about feature selection methodology.
7. Model Training & Evaluation
Sequential usage instructions for:
•	Training individual models (RF, XGBoost, LightGBM)
•	Applying max voting ensemble aggregation
•	Computing performance metrics (accuracy, precision, recall, F1-score per class)
•	Expected results table showing class-specific performance
8. XAI Methods Documentation
A dedicated section describing each of the seven XAI methods:
•	Grad-CAM: Gradient-weighted class activation mapping for convolutional layers
•	Integrated Gradients: Attribution via path integration from baseline to input
•	Saliency Maps: Gradient-based pixel importance
•	SmoothGrad: Noise-averaging for more stable attributions
•	LIME: Local linear approximations with superpixel segmentation
•	DeepLIFT: Reference-based feature importance scores
•	SHAP: Shapley value-based explanations with gradient explainer
Each method includes interpretation guidance (red/yellow = high attention, blue = low attention, clinical implications).
9. Running the Full Pipeline
A single bash command or step-by-step sequential execution instructions that exactly reproduce all reported results, including:
•	Confusion matrix (Figure X)
•	Class-specific metrics table (Table 5)
•	Comprehensive XAI comparison visualizations
•	Summary analysis report
10. Troubleshooting & Known Issues
Common problems (memory requirements for LightGBM >8GB, Kaggle API setup, GA convergence tuning) with explicit solutions.


