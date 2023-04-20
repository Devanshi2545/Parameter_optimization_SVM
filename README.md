# Parameter-Optimisation-SVM


## EDA for Dataset
- Exploring the dataset, getting summary statistics and checking for null values and duplicates and there weren't any.
- Graphical representations:\
1- Count plot the labels column show the distribution of all classes that showed a slight imbalance but it doesn't affect and no need to handle.\
2- Histogram of numerical features, some distributions have long tails, skewed and most are bi-modal which means that some classes are quite distinct from others.
\
3- Boxplot shows that the "Bombay" & "Horoz" classes are distinct from other classes and that there are some minimal outliers in some features.\
4- The Pearson linear correlation shows that there are lots of highly correlated features.


**Dataset Used:** [Dry bean Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)

| Number of Instances:  | 13000  |
|-----------------------|--------|
| Number of Attributes: | 17     |

---

**Parameter Grid Used**
---
|Hyperparameter         |Values                |
|-----------------------|----------------------|
| kernel                | 'linear', 'poly', 'rbf', 'sigmoid' |
| C                     | 0.001, 0.01, 0.1, 1, 10    |
| gamma                 |['scale', 'auto'] + list(np.logspace(-3, 3, 7))   |

---

## EDA
- Exploring the dataset, getting summary statistics and checking for null values and duplicates and there weren't any.
- Graphical representations:\
1- Count plot the labels column show the distribution of all classes that showed a slight imbalance but it doesn't affect and no need to handle.\
2- Histogram of numerical features, some distributions have long tails, skewed and most are bi-modal which means that some classes are quite distinct from others.
\
3- Boxplot shows that the "Bombay" & "Horoz" classes are distinct from other classes and that there are some minimal outliers in some features.\
4- The Pearson linear correlation shows that there are lots of highly correlated features.
 
 | Sample Number | Best Accuracy | Kernel | C  | gamma |
|----------|---------------|--------|-----|-------|
| 1        | 0.799       | rbf    | 1.000 | 0.001   |
| 2        | 0.765        |  poly    | 0.001 | scale   |
| 3        | 0.801        | rbf    | 10.000 | 0.01   |
| 4        | 0.801      | rbf    | 1.000 | auto   |
| 5        | 0.811       | rbf    | 1.000 | 0.1   |
| 6        | 0.799        | rbf    | 1.000 | 0.001  |
| 7        | 0.780        | rbf    | 1.000 | 0.1   |
| 8        | 0.799        | rbf    | 1.000 | 0.1   |
| 9        | 0.770        | rbf    | 1.000 | 0.1   |
| 10       | 0.784       | rbf    | 1.000 | 0.001 |

---

**Sample 5 gives the Best SVM accuracy with params: rbf,1.000,0.1 for Kernel,C and Gamma respectively**

---

Graph of Accuracy per 1000 iterations for Sample 5:

![alt text](https://github.com/UTK21/Parameter-optimisation-SVM/blob/main/screenshot.png)
