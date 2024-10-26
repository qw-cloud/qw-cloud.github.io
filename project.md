## Final Project: Automated Aircraft Detection in Satellite Imagery Using Deep Learning
Frank Wu

Department of Environmental Health Science, UCLA

AOS C204: Introduction to Machine Learning for the Physical Sciences

Dr. Alexander Lozinski

December 2024

***

## Introduction 

Satellite imagery provides insights across sectors, including agriculture, defense intelligence, energy, and financial services. The advancement of satellite imaging technology has generated an exponential increase in daily image acquisition, rendering manual analysis computationally intractable. This technological evolution necessitates the implementation of automated machine learning and computer vision algorithms for efficient processing and analysis of massive image datasets.
This investigation proposes a deep learning framework utilizing convolutional neural networks (CNNs) for automated aircraft detection and recognition in satellite imagery. The methodology supports airport monitoring, traffic flow analysis, and defense intelligence applications. The implementation demonstrates improvements in detection efficiency while  reducing manual monitoring costs, establishing a foundation for automated GIS analysis.

## Data

This investigation utilizes the ["Planes in Satellite Imagery"](https://www.kaggle.com/datasets/rhammell/planesnet/data) dataset, derived from [Planet Labs](https://www.planet.com/)' commercial satellite imagery, encompassing satellite captures over multiple airports in California, United States. The dataset comprises 32,000 RGB images, each sized at 20x20 pixels, categorized into binary classes: "plane" and "no-plane", facilitating machine learning model development for aircraft detection and localization in satellite imagery.
Dataset Composition:

**Positive Class (Aircraft Present):**

Sample size: 8,000 images

Characteristics: Aircraft fuselage centered in frame

Features: Distinct wing structures, tail assemblies, and nose sections

Variability: Multiple aircraft configurations, orientations, and atmospheric conditions

![image](https://github.com/user-attachments/assets/c4ec1408-b439-4c46-94e6-9ea5c1f5e8b3)
**Source**: ["Planes in Satellite Imagery"](https://www.kaggle.com/datasets/rhammell/planesnet/data)

**Negative Class (Aircraft Absent):**

Sample size: 24,000 images

Subcategories:

a) Complete absence of aircraft (terrain features, water bodies, vegetation, built structures)

b) Partial aircraft features

c) Misclassified instances due to high-intensity reflectance or linear terrain features


**Annotation Specifications:**

Binary classification labels (1: aircraft present, 0: aircraft absent)

Scene identifiers

Geocentric coordinates (latitude/longitude) for image centroids

**Validation Framework:**

Primary validation: Four complete satellite scenes provided within the dataset

Supplementary validation: Four independent airport scenes acquired through external sources for robustness assessment 

The dataset structure follows established principles in remote sensing research: 

The 1:3 class distribution ratio aligns with standard practices in computer vision for handling class imbalance
The image resolution and dimensionality correspond to optimal parameters identified in satellite-based object detection literature
The inclusion of diverse environmental conditions enhances model generalization capabilities

This structured approach to dataset implementation enables systematic evaluation of model performance across varied operational conditions, while the supplementary validation framework provides additional assessment of model robustness and generalization capacity.

![image](https://github.com/user-attachments/assets/037c60e2-a16e-4492-9311-04f408902bb3)
**Source**: ["Planes in Satellite Imagery"](https://www.kaggle.com/datasets/rhammell/planesnet/data)


## Preliminary Data Visualization


The top row of the first figure (with aircraft present) includes three heatmaps representing the mean red, green, and blue (RGB) values from 8,000 images containing aircraft. These heatmaps show higher RGB values concentrated in the center, with a distinct transition from red to yellow, indicating elevated RGB levels in central regions. This pattern arises because aircraft are generally positioned near the center, resulting in higher average RGB values in a cross-like shape. In contrast, the bottom row (without aircraft) displays heatmaps with a more uniform color distribution and smaller differences between the center and edges, reflecting the more consistent RGB distribution across the 24,000 aircraft-free images. In these images, overall RGB levels are lower, particularly within the red and green channels.

![rgb_analysis](https://github.com/user-attachments/assets/75105ec3-78cd-44a7-bc84-be447639cf11)

**Figure 1: Average RGB values**


**Modelling**

Initially, several classical algorithms were applied to the aircraft recognition task. A logistic regression model, set with a maximum of 1000 iterations, achieved an accuracy of 90.80%, though with a relatively long training time of 4.51 seconds. A random forest classifier, configured with 100 estimators and parallel computation enabled, demonstrated robust classification performance with an accuracy of 95.08% and a precision of 93.01%. The XGBoost model, also using 100 tree estimators, with a learning rate of 0.1 and a maximum depth of 6, performed best among traditional machine learning models, reaching an accuracy of 95.73% using histogram-based optimization. LightGBM, with parameters similar to XGBoost, achieved a comparable accuracy of 95.55%. The support vector machine (SVM) with an RBF kernel attained an accuracy of 95.72% but exhibited the longest training time, at 105.70 seconds. The k-nearest neighbors classifier, configured with 5 neighbors, yielded an accuracy of 94.75% and a recall of 91.60%. The decision tree and Naïve Bayes models achieved accuracies of 89.72% and 63.98%, respectively, with Naïve Bayes performing fastest (0.43 seconds) but less effectively overall.

![model_comparison_light](https://github.com/user-attachments/assets/720d8438-64f1-4582-ac63-d2c240b38ff0)
**Figure 2: Performance of Classic Machine Learning**

For deep learning models, three convolutional neural network (CNN) architectures were implemented and evaluated. The basic CNN model employed a simple design, comprising two convolutional layers (with 32 and 64 filters) combined with max pooling, followed by fully connected layers for output, achieving an accuracy of 97.02%, precision of 92.39%, and recall of 95.92%. A deeper CNN model, with a more complex architecture incorporating batch normalization and dropout layers (rates of 0.25 and 0.5), along with additional convolutional filters (64 and 128), reached an accuracy of 90.20% and a precision of 99.79%, though with a lower recall of 60.82%. The custom ResNet model, featuring residual connections and residual blocks with 32 and 64 filters, combined with batch normalization and global average pooling layers, achieved the best overall performance across all models, with an accuracy of 98.62%, precision of 97.36%, recall of 97.12%, and F1 score of 97.24%. All deep learning models used the Adam optimizer and binary cross-entropy loss function, training over 20 epochs with a batch size of 32 and a validation split of 0.2.

![model_comparison](https://github.com/user-attachments/assets/16063914-f2f8-4582-82c7-31891351d9ff)

**Figure 3: Performance of Deep Learning**

These experimental results clearly indicate that deep learning methods, particularly those incorporating residual connections, exhibit superior performance in this image recognition task. Although certain traditional machine learning methods (such as XGBoost and SVM) achieved relatively high performance, the deep learning models demonstrated enhanced feature extraction and classification capabilities, especially when considering accuracy, precision, and recall comprehensively.


## Results




The custom ResNet model demonstrated exceptional classification performance, as evidenced by its confusion matrix: true negatives (TN) totaled 23,796 (99.2%), true positives (TP) reached 7,665 (95.8%), false negatives (FN) were 335 (4.2%), and false positives (FP) remained low at 204 (0.9%). These results indicate high model accuracy in distinguishing positive and negative samples, with particularly strong control over the false positive rate.

![confusion_matrix_enhanced](https://github.com/user-attachments/assets/9c56290a-a76a-49ad-a760-d832e6237680)
**Figure 4: ResNet Confusion Matrix**

The model’s classification strength is further highlighted by an area under the ROC curve (AUC) of 0.997, close to the ideal value of 1.0, demonstrating excellent discriminative ability. Additionally, the precision-recall (PR) curve yielded an average precision (AP) of 0.990, with a consistently high curve, indicating that the model sustains both high precision and high recall.

![image](https://github.com/user-attachments/assets/7111eb5a-eb7e-4604-9b08-20feb1cdd746)
**Figure 5: ResNet Performance Curves**


The model's training history further supports its strong learning characteristics:

**Accuracy**: Training accuracy (solid blue line) steadily increased from an initial 0.93 to exceed 0.99, while validation accuracy (dashed blue line) showed some variability but generally maintained high values (between 0.85 and 0.98).

**Loss**: Training loss (solid red line) consistently decreased to a low level, while validation loss (dashed red line) displayed occasional fluctuations (notably around epochs 5 and 15) but trended favorably overall without signs of persistent overfitting.
![image](https://github.com/user-attachments/assets/f4949e50-213a-445a-a3fa-83231c701169)
**Figure 6: ResNet Training History**

These metrics collectively suggest that the custom ResNet model achieved outstanding classification performance with robust generalization. While some fluctuation occurred during training, the model ultimately stabilized, achieving effective feature learning likely enhanced by the unique advantages of residual connections in handling image features.

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)


