from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tools.email_preprocess import preprocess

words_file = "/word_data.pkl"
authors_file = "/email_authors.pkl"

features_train, features_test, labels_train, labels_test = preprocess(words_file, authors_file)



# NOTE: The default value of gamma changed from "auto" to "scale"
clf = SVC(kernel='rbf', C=10000, gamma='auto')

print("Started Training the model!")
time_fit = time()
clf.fit(features_train, labels_train)
print("Finished Training the model!")
time_fit_end = time()
print(f"Training time was: {time_fit_end - time_fit}")

time_predict = time()
pred = clf.predict(features_test)
time_predict_end = time()

print(f"Predicting time was: {time_predict_end - time_predict}")

print(f"Model score: {accuracy_score(pred, labels_test)}")
