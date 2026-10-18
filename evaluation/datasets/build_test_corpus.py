"""Builds the controlled evaluation corpus: three small PDFs, one topic per page.

The texts are written for this project, so ground truth is fully known and the
evaluation dataset (qa_dataset.json) can point at exact (file, page) evidence.

Run:  python evaluation/datasets/build_test_corpus.py
"""
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent / "corpus"

CORPUS = {
    "data_science_regression.pdf": [
        "Simple linear regression. Simple linear regression models the relationship between one predictor x and a numeric "
        "response y with the line y = b0 + b1*x + error. The coefficients are estimated by ordinary least squares, which "
        "chooses the line that minimises the sum of squared residuals. A residual is the difference between an observed value "
        "and the value predicted by the line. The coefficient of determination, R-squared, is the proportion of variance in the "
        "response explained by the model and always lies between 0 and 1 for a model with an intercept. Simple regression has "
        "exactly one predictor, so it cannot separate the effects of several correlated variables.",
        "Multiple regression. Multiple regression extends the model to several predictors: y = b0 + b1*x1 + b2*x2 + ... + bk*xk "
        "+ error. Each coefficient is interpreted as the expected change in the response for a one unit increase in that "
        "predictor while all other predictors are held constant. Adding predictors never decreases R-squared, so adjusted "
        "R-squared is used because it penalises predictors that add no explanatory power. Multicollinearity occurs when "
        "predictors are strongly correlated with each other; it inflates the standard errors of the coefficients. The variance "
        "inflation factor, VIF, quantifies it, and a VIF above 10 is a common warning sign.",
        "Overfitting and regularisation. A model overfits when it fits noise in the training data and performs worse on new data. "
        "The bias-variance trade-off describes how simple models have high bias and low variance while flexible models have low "
        "bias and high variance. Ridge regression adds an L2 penalty equal to the sum of squared coefficients, which shrinks "
        "coefficients towards zero but rarely makes them exactly zero. Lasso regression adds an L1 penalty equal to the sum of "
        "absolute coefficients, which can set some coefficients exactly to zero and therefore performs feature selection. K-fold "
        "cross-validation estimates generalisation error by training on k-1 folds and testing on the remaining fold, repeating "
        "k times.",
    ],
    "ml_clustering_classification.pdf": [
        "K-means clustering. K-means partitions n observations into k clusters by repeatedly assigning each point to the nearest "
        "centroid and then recomputing every centroid as the mean of its assigned points. The algorithm minimises the "
        "within-cluster sum of squares and stops when assignments no longer change. The number of clusters k must be chosen in "
        "advance. The elbow method plots the within-cluster sum of squares against k and looks for the point where the "
        "improvement flattens. The silhouette score compares how close a point is to its own cluster with how close it is to the "
        "nearest other cluster, with values near 1 indicating well separated clusters. K-means is sensitive to the initial "
        "centroids, so it is usually run several times.",
        "Logistic regression and classification metrics. Logistic regression predicts the probability of a binary outcome by "
        "applying the sigmoid function to a linear combination of predictors. A decision threshold, commonly 0.5, converts the "
        "probability into a class label. Precision is the fraction of predicted positives that are truly positive. Recall is the "
        "fraction of true positives that the model finds. The F1 score is the harmonic mean of precision and recall. Lowering the "
        "threshold usually raises recall and lowers precision. Accuracy can be misleading when classes are imbalanced.",
        "Decision trees and random forests. A decision tree splits the data using feature thresholds chosen to reduce impurity; "
        "the Gini impurity of a node is one minus the sum of squared class proportions. Deep trees overfit easily. A random "
        "forest trains many trees on bootstrap samples of the data and considers only a random subset of features at each "
        "split, then averages their predictions. This bagging approach reduces variance without greatly increasing bias. The "
        "out-of-bag error provides a built-in estimate of test error.",
    ],
    "statistics_inference.pdf": [
        "Hypothesis testing. A hypothesis test starts with a null hypothesis, usually stating no effect or no difference. The "
        "p-value is the probability of observing data at least as extreme as the sample if the null hypothesis were true; it is "
        "not the probability that the null hypothesis is true. A result is called statistically significant when the p-value is "
        "below the chosen significance level alpha, commonly 0.05. A type I error rejects a true null hypothesis and occurs with "
        "probability alpha. A type II error fails to reject a false null hypothesis; the power of a test is one minus the "
        "probability of a type II error.",
        "Confidence intervals and the central limit theorem. The central limit theorem states that the distribution of the "
        "sample mean approaches a normal distribution as the sample size grows, whatever the shape of the population "
        "distribution, provided the variance is finite. A 95 percent confidence interval is built by a procedure that captures "
        "the true parameter in 95 percent of repeated samples. It does not mean there is a 95 percent probability that a "
        "specific computed interval contains the parameter. Larger samples produce narrower intervals because the standard "
        "error of the mean equals the standard deviation divided by the square root of n.",
        "ANOVA and model comparison. One-way ANOVA tests whether the means of three or more groups are equal by comparing "
        "between-group variance with within-group variance through the F-statistic. A large F-statistic with a small p-value "
        "suggests at least one group mean differs. ANOVA does not say which groups differ, so a post hoc test such as Tukey's "
        "HSD is used. For comparing regression models, AIC and BIC reward goodness of fit but penalise the number of "
        "parameters; BIC applies a heavier penalty than AIC when the sample is large. RMSE is the square root of the mean "
        "squared prediction error and is in the same units as the response.",
    ],
}


def build(out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, pages in CORPUS.items():
        doc = pymupdf.open()
        for text in pages:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(60, 60, 540, 780), text, fontsize=11)
        path = out / name
        doc.save(path)
        doc.close()
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in build():
        print("wrote", p)
