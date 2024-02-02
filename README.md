# Thesis Project

The objective of this project is to evaluate the feasibility of one-class classifiers in fault-detection task for sanding machines using recorded sound samples. The repository includes Python code for conducting all preliminary analyses.

# Getting Started

### Installation
1. Clone the repository.
1. Modify the paths in Code1/my_configuration.py to match your settings.
1. Adjust project parameters in Code1/my_configuration.py, e.g., the frequency resolution (OPTIONAL).
1. run extract_spectral_data.py

### Analyses
Look at the example notebooks.
* **Zub_Data_extraction.ipynb** highlights the idea behind the data extraction.
* **Anomaly detection.ipynb** examples of anomaly detection methods.

### Prerequisites
The required packages are most easily installed through Anaconda (all are included in the base environment for the full installation, i.e., not miniconda).
* numpy
* matplotlib
* scipy
* sklearn
* pandas
* iPython
* Jupyter notebook

### Machine information
Original machines (EPS project Spring 2021). The Machine number is written on the side.

| Machine ID | Comment | State  | Fault ID | Fault description|
| - | - | - | - | - |
| 1. | Good | OK | 0 | - |
| 2. | Good | OK | 0 | - |
| 3. | Uneven plate | Faulty | 1 | Uneven plate |
| 4. | Fan problem | Faulty | 2 | Fan problem |
| 5. | Fan missound | Faulty | 2 | Fan problem |
| 6. | Fan missound | Faulty | 2 | Fan problem |
| 7. | Motor bearing | Faulty | 3 | Bearing problem |
| 8. | Bearing missound (lower speed)) | Faulty | 3 | Bearing problem |
| 9. | Bearing problem | Faulty | 3 | Bearing problem |
| 10. | Spider bearing | Faulty | 4 | Spider fault |

Additional machines (received 14.9.2021). The Machine number is written on the top side next to the power button.

| Machine ID | Comment | State  | Fault ID | Fault description|
| - | - | - | - | - |
| 1. | Gen. 2 motor | OK | 0 | - |
| 2. | Gen. 2 motor | OK | 0 | - |
| 3. | Gen. 2 motor | OK | 0 | - |
| 4. | - | OK | 0 | - |
| 5. | - | OK | 0 | - |
| 6. | - | OK | 0 | - |
| 7. | - | OK | 0 | - |
| 8. | - | OK | 0 | - |
| 9. | - | OK | 0 | - |
| 10. | - | OK | 0 | - |
| 11. | - | OK | 0 | - |
| 12. | Brakeseal/fan sound | Faulty | 2 | Fan problem |
| 13. | Spindle bearing| Faulty | 5 | Spindle bearing |
| 14. | - | OK | 0 | - |
| 15. | - | OK | 0 | - |
| 16. | - | OK | 0 | - |
| 17. | - | OK | 0 | - |
| 18. | - | OK | 0 | - |
| 19. | - | OK | 0 | - |
| 20. | - | OK | 0 | - |

### Analysis comments
The results of the experiments reveal that the one-class SVM fails to detect anomalous data 
in this particular case. This could be attributed to the unsuitability of the data distribution 
for distinguishing anomalies using a hyperplane, even with the utilization of kernel tricks. 
The one-class SVM considers the origin as anomalous data and draws a hyperplane 
between the normal data and the origin. However, in this experiment, there is no 
guarantee that the anomalous data are located close to the origin. Consequently, the one class SVM completely fails to detect anomalous data in this case.
Furthermore, although the performance of the isolation forest is better than that of the 
one-class SVM, it is still not adequate for practical utilization of the model. The isolation 
forest comprises several isolation trees that consider features randomly. Since the model 
was trained only with normal data in this experiment, the isolation trees were unable to 
identify the most relevant features from the normal data for anomaly detection. 
Consequently, when the model is tested with anomalous data, the result from one isolation 
tree may be mitigated by the result from another isolation tree. Therefore, if our 
anomalous data exhibit variation in fewer features, the results from fewer isolation trees 
may be mitigated as well. Thus, the isolation forest is suitable for cases where the training 
data contains anomalous data.
However, this experiment also demonstrates that the autoencoder performs well in 
anomaly detection in this particular case. Additionally, nearest neighbor-based 
classification methods such as k-nearest neighbors (KNN) and LOF yield satisfactory results. 
Another noteworthy finding is that dimensionality reduction using PCA enhances the 
performance of the k-nearest neighbor algorithm.
From this experiment, it is observed that LOF achieves the best classification results 
compared to other machine learning algorithms examined. Another significant finding is 
that while dimensionality reduction enhances the performance of LOF in the validation 
phase, it negatively affects its performance in the testing phase. This discrepancy can be 
attributed to overfitting, which can be mitigated by adjusting the n-components of PCA.
In fact, nearest neighbor-based one-class classifiers such as KNN, LOF, and PCA-based 
autoencoder prioritize every feature during classification. Consequently, even a slight 
variation in any single feature can lead to a significant difference in the anomaly scores. In 
this experiment, since the sound from faulty machines can cause changes in any feature (frequency), the important features for anomaly detection remain uncertain. Thus, classification algorithms that assign importance to every feature, like nearest neighbor based algorithms and PCA-based autoencoder, yield better results.
Analysing the anomaly scores for LOF, it is evident that machine numbers 6, 9, and 23 
exhibit significantly higher scores. Machine number 6 experiences a fan problem, machine 
number 9 has a bearing problem, and machine number 23 suffers from a spindle bearing 
issue. Machine numbers 7 and 8, both having the same bearing problem, produce almost 
identical anomaly scores, slightly higher than those of the normal machines. Conversely, 
machine number 10, which encounters a spider bearing problem, generates anomaly 
scores similar to those of the normal machines

### Data comments
Data has been collected in five steps:

* **recordings_22_02_21:** 1 x 30 s, all 30 machines, RPM 4k and 10k, tag z, machines in a fixed position.
* **recordings_22_02_23:** 1 x 30 s, all 30 machines, RPM 4k and 10k, tag z2, machines in a fixed position.
* **recordings_22_03_02:** 2 x 30 s, 20 new machines, RPM 4k, tag z. machines moved randomly.
* **recordings_22_03_16:** 4 x 20 s, all 30 machines, RPM 4k, tag z, machines moved randomly.
* **recordings_22_03_29:** 4 x 20 s, all 30 machines, RPM 4k, tag z2, machines moved randomly.

