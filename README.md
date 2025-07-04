# DECISION-TREE-IMPLEMENTATION

*COMPANY* : CODTECH  IT SOLUTIONS

*NAME* : SAITEJA BADDIREDDY

*INTERN ID* : :CT08DM66

*DOMAIN* : MACHINE LEARNING

*DURATION* : 8 WEEKS

*MENTOR* : NEELA SANTOSH

#DESCRIPTION:

Importing Libraries: The code begins by importing essential libraries:

pandas and numpy for data manipulation and numerical operations.
matplotlib.pyplot and seaborn for data visualization.
sklearn modules for machine learning functionalities, including model training, evaluation, and hyperparameter tuning.
Loading the Dataset: The Iris dataset is loaded using load_iris() from sklearn.datasets. The features are stored in a DataFrame X, while the target labels (species) are stored in a Series y. The feature names and target names are also extracted for later use in visualizations.

Data Splitting: The dataset is split into training and testing sets using train_test_split(). This function ensures that 70% of the data is used for training the model and 30% for testing. The stratify parameter is set to y to maintain the proportion of each class in both the training and testing sets, which is crucial for classification tasks.

Hyperparameter Tuning: The code employs GridSearchCV to perform hyperparameter tuning. A grid of parameters is defined, including:

criterion: The function to measure the quality of a split (options are 'gini' and 'entropy').
max_depth: The maximum depth of the tree, which helps prevent overfitting.
min_samples_split: The minimum number of samples required to split an internal node.
min_samples_leaf: The minimum number of samples required to be at a leaf node.
The GridSearchCV object is created with a 5-fold cross-validation strategy, and it fits the model to the training data. After fitting, the best parameters are printed, and the best estimator is stored in clf.

Visualization of the Decision Tree: The optimized decision tree is visualized using plot_tree(). This function provides a graphical representation of the tree structure, showing how the model makes decisions based on the input features. The tree is filled with colors representing different classes, making it easier to interpret.

Making Predictions: The trained model is used to make predictions on the test set using the predict() method. The predicted labels are stored in y_pred.

Model Evaluation: The accuracy of the model is calculated using accuracy_score(), which compares the predicted labels with the actual labels from the test set. The accuracy score is printed to provide a quick assessment of the model's performance.

Confusion Matrix: A confusion matrix is generated using confusion_matrix(), which provides a detailed breakdown of the model's performance across different classes. This matrix is visualized using a heatmap from the seaborn library, allowing for an intuitive understanding of where the model is making correct and incorrect predictions.

Classification Report: The classification report is generated using classification_report(), which includes precision, recall, and F1-score for each class. This report provides a more nuanced view of the model's performance, especially in cases where class distributions are imbalanced.

Feature Importances: Finally, the feature importances are extracted from the trained model using clf.feature_importances_. This provides insight into which features are most influential in making predictions. A horizontal bar plot is created to visualize the importance scores of each feature, helping to understand the model's decision-making process.


#OUTPUT

![Image](https://github.com/user-attachments/assets/3590b62c-6c16-4d93-9c32-028e3db31961)
