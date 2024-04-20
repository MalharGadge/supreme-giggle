from sklearn.neighbors import KNeighborsClassifier

# Load the trained KNN classifier
knn_classifier = KNeighborsClassifier()
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

cv_scores = cross_val_score(knn_classifier, features_pca, labels, cv=kf)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))
