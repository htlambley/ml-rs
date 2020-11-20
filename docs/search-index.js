var searchIndex = JSON.parse('{\
"ml_rs":{"doc":"","i":[[4,"Error","ml_rs","",null,null],[13,"UseBeforeFit","","",0,null],[13,"InvalidTrainingData","","",0,null],[13,"OptimiserError","","",0,null],[13,"FittingError","","",0,null],[0,"classification","","This module contains a variety of classification models to…",null,null],[3,"TrivialClassifier","ml_rs::classification","A trivial classifier that is initialised with a class…",null,null],[3,"MajorityClassifier","","A classifier which learns the most common class and…",null,null],[5,"labels_binary","","Convenience function to verify whether an array of labels…",null,[[["arrayview1",6]]]],[0,"linear","","Classifiers based on linear regression (which, despite its…",null,null],[3,"IRLSLogisticRegression","ml_rs::classification::linear","A classifier implementing the logistic regression model…",null,null],[12,"max_iter","","The maximum number of iterations of the iteratively…",1,null],[3,"LogisticRegression","","A classifier implementing the logistic regression model.…",null,null],[12,"max_iter","","The maximum number of iterations of the BFGS algorithm to…",2,null],[11,"new","","Creates a new `IRLSLogisticRegression` classifier without…",1,[[],["irlslogisticregression",3]]],[11,"new","","Creates a new `LogisticRegression` classifier which must…",2,[[],["logisticregression",3]]],[8,"Classifier","ml_rs::classification","This trait represents a classifier that can be fit on…",null,null],[10,"fit","","Fits the classifier to the given data matrix `x` and…",3,[[["arrayview2",6],["arrayview1",6]],[["result",4],["error",4]]]],[10,"predict","","Makes a prediction for the sample in each row of the data…",3,[[["arrayview2",6]],[["array1",6],["result",4],["error",4]]]],[8,"ProbabilityBinaryClassifier","","A binary classifier that can return calibrated probability…",null,null],[10,"predict_probability","","Makes an estimate of the probability that each sample in…",4,[[["arrayview2",6]],[["result",4],["error",4],["array1",6]]]],[11,"new","","Creates a new `TrivialClassifier` which will always return…",5,[[],["trivialclassifier",3]]],[11,"new","","Creates a new `MajorityClassifier` ready to be fit on the…",6,[[],["majorityclassifier",3]]],[0,"metrics","ml_rs","A collection of metrics to measure the performance of…",null,null],[5,"accuracy_score","ml_rs::metrics","Calculate the accuracy of an array of predictions `y_pred`…",null,[[["arrayview1",6]]]],[0,"binary","","",null,null],[5,"precision_recall_score","ml_rs::metrics::binary","Calculates the precision and recall of a binary classifier…",null,[[["arrayview1",6]]]],[0,"preprocessing","ml_rs","",null,null],[3,"CsvReader","ml_rs::preprocessing","",null,null],[11,"new","","",7,[[["file",3]],["csvreader",3]]],[11,"read","","",7,[[],[["sized",8],["array2",6],["deserialize",8]]]],[0,"regression","ml_rs","",null,null],[0,"linear","ml_rs::regression","",null,null],[3,"LinearRegression","ml_rs::regression::linear","Fits a linear model to data using the ordinary least…",null,null],[11,"new","","",8,[[],["linearregression",3]]],[8,"Regressor","ml_rs::regression","This trait represents a regression model, used to predict…",null,null],[10,"fit","","",9,[[["arrayview2",6],["arrayview1",6]],[["result",4],["error",4]]]],[10,"predict","","",9,[[["arrayview2",6]],[["result",4],["error",4],["array1",6]]]],[0,"transformation","ml_rs","",null,null],[0,"pca","ml_rs::transformation","",null,null],[3,"PrincipalComponentAnalysis","ml_rs::transformation::pca","A transformer that can perform principal component…",null,null],[11,"new","","",10,[[],["principalcomponentanalysis",3]]],[8,"Transformer","ml_rs::transformation","This trait represents a transformer, which performs some…",null,null],[10,"fit","","",11,[[["arrayview2",6]],[["result",4],["error",4]]]],[10,"transform","","",11,[[["arrayview2",6]],[["result",4],["array2",6],["error",4]]]],[11,"fit_transform","","",11,[[["arrayview2",6]],[["result",4],["array2",6],["error",4]]]],[11,"from","ml_rs","",0,[[]]],[11,"into","","",0,[[]]],[11,"to_owned","","",0,[[]]],[11,"clone_into","","",0,[[]]],[11,"to_string","","",0,[[],["string",3]]],[11,"try_from","","",0,[[],["result",4]]],[11,"try_into","","",0,[[],["result",4]]],[11,"borrow","","",0,[[]]],[11,"borrow_mut","","",0,[[]]],[11,"type_id","","",0,[[],["typeid",3]]],[11,"init","","",0,[[]]],[11,"deref","","",0,[[]]],[11,"deref_mut","","",0,[[]]],[11,"drop","","",0,[[]]],[11,"vzip","","",0,[[]]],[11,"from","ml_rs::classification","",5,[[]]],[11,"into","","",5,[[]]],[11,"to_owned","","",5,[[]]],[11,"clone_into","","",5,[[]]],[11,"try_from","","",5,[[],["result",4]]],[11,"try_into","","",5,[[],["result",4]]],[11,"borrow","","",5,[[]]],[11,"borrow_mut","","",5,[[]]],[11,"type_id","","",5,[[],["typeid",3]]],[11,"init","","",5,[[]]],[11,"deref","","",5,[[]]],[11,"deref_mut","","",5,[[]]],[11,"drop","","",5,[[]]],[11,"vzip","","",5,[[]]],[11,"from","","",6,[[]]],[11,"into","","",6,[[]]],[11,"to_owned","","",6,[[]]],[11,"clone_into","","",6,[[]]],[11,"try_from","","",6,[[],["result",4]]],[11,"try_into","","",6,[[],["result",4]]],[11,"borrow","","",6,[[]]],[11,"borrow_mut","","",6,[[]]],[11,"type_id","","",6,[[],["typeid",3]]],[11,"init","","",6,[[]]],[11,"deref","","",6,[[]]],[11,"deref_mut","","",6,[[]]],[11,"drop","","",6,[[]]],[11,"vzip","","",6,[[]]],[11,"from","ml_rs::classification::linear","",1,[[]]],[11,"into","","",1,[[]]],[11,"try_from","","",1,[[],["result",4]]],[11,"try_into","","",1,[[],["result",4]]],[11,"borrow","","",1,[[]]],[11,"borrow_mut","","",1,[[]]],[11,"type_id","","",1,[[],["typeid",3]]],[11,"init","","",1,[[]]],[11,"deref","","",1,[[]]],[11,"deref_mut","","",1,[[]]],[11,"drop","","",1,[[]]],[11,"vzip","","",1,[[]]],[11,"from","","",2,[[]]],[11,"into","","",2,[[]]],[11,"to_owned","","",2,[[]]],[11,"clone_into","","",2,[[]]],[11,"try_from","","",2,[[],["result",4]]],[11,"try_into","","",2,[[],["result",4]]],[11,"borrow","","",2,[[]]],[11,"borrow_mut","","",2,[[]]],[11,"type_id","","",2,[[],["typeid",3]]],[11,"init","","",2,[[]]],[11,"deref","","",2,[[]]],[11,"deref_mut","","",2,[[]]],[11,"drop","","",2,[[]]],[11,"vzip","","",2,[[]]],[11,"from","ml_rs::preprocessing","",7,[[]]],[11,"into","","",7,[[]]],[11,"try_from","","",7,[[],["result",4]]],[11,"try_into","","",7,[[],["result",4]]],[11,"borrow","","",7,[[]]],[11,"borrow_mut","","",7,[[]]],[11,"type_id","","",7,[[],["typeid",3]]],[11,"init","","",7,[[]]],[11,"deref","","",7,[[]]],[11,"deref_mut","","",7,[[]]],[11,"drop","","",7,[[]]],[11,"vzip","","",7,[[]]],[11,"from","ml_rs::regression::linear","",8,[[]]],[11,"into","","",8,[[]]],[11,"to_owned","","",8,[[]]],[11,"clone_into","","",8,[[]]],[11,"try_from","","",8,[[],["result",4]]],[11,"try_into","","",8,[[],["result",4]]],[11,"borrow","","",8,[[]]],[11,"borrow_mut","","",8,[[]]],[11,"type_id","","",8,[[],["typeid",3]]],[11,"init","","",8,[[]]],[11,"deref","","",8,[[]]],[11,"deref_mut","","",8,[[]]],[11,"drop","","",8,[[]]],[11,"vzip","","",8,[[]]],[11,"from","ml_rs::transformation::pca","",10,[[]]],[11,"into","","",10,[[]]],[11,"to_owned","","",10,[[]]],[11,"clone_into","","",10,[[]]],[11,"try_from","","",10,[[],["result",4]]],[11,"try_into","","",10,[[],["result",4]]],[11,"borrow","","",10,[[]]],[11,"borrow_mut","","",10,[[]]],[11,"type_id","","",10,[[],["typeid",3]]],[11,"init","","",10,[[]]],[11,"deref","","",10,[[]]],[11,"deref_mut","","",10,[[]]],[11,"drop","","",10,[[]]],[11,"vzip","","",10,[[]]],[11,"fit","ml_rs::classification::linear","",1,[[["arrayview2",6],["arrayview1",6]],[["result",4],["error",4]]]],[11,"predict","","",1,[[["arrayview2",6]],[["array1",6],["result",4],["error",4]]]],[11,"fit","","",2,[[["arrayview2",6],["arrayview1",6]],[["result",4],["error",4]]]],[11,"predict","","",2,[[["arrayview2",6]],[["array1",6],["result",4],["error",4]]]],[11,"fit","ml_rs::classification","",5,[[["arrayview2",6],["arrayview1",6]],[["result",4],["error",4]]]],[11,"predict","","",5,[[["arrayview2",6]],[["array1",6],["result",4],["error",4]]]],[11,"fit","","",6,[[["arrayview2",6],["arrayview1",6]],[["result",4],["error",4]]]],[11,"predict","","",6,[[["arrayview2",6]],[["array1",6],["result",4],["error",4]]]],[11,"predict_probability","ml_rs::classification::linear","",1,[[["arrayview2",6]],[["result",4],["error",4],["array1",6]]]],[11,"predict_probability","","",2,[[["arrayview2",6]],[["result",4],["error",4],["array1",6]]]],[11,"fit","ml_rs::regression::linear","",8,[[["arrayview2",6],["arrayview1",6]],[["result",4],["error",4]]]],[11,"predict","","",8,[[["arrayview2",6]],[["result",4],["error",4],["array1",6]]]],[11,"fit","ml_rs::transformation::pca","",10,[[["arrayview2",6]],[["result",4],["error",4]]]],[11,"transform","","",10,[[["arrayview2",6]],[["result",4],["array2",6],["error",4]]]],[11,"clone","ml_rs::classification::linear","",2,[[],["logisticregression",3]]],[11,"clone","ml_rs::classification","",5,[[],["trivialclassifier",3]]],[11,"clone","","",6,[[],["majorityclassifier",3]]],[11,"clone","ml_rs::regression::linear","",8,[[],["linearregression",3]]],[11,"clone","ml_rs::transformation::pca","",10,[[],["principalcomponentanalysis",3]]],[11,"clone","ml_rs","",0,[[],["error",4]]],[11,"default","ml_rs::classification::linear","",1,[[],["irlslogisticregression",3]]],[11,"default","","",2,[[],["logisticregression",3]]],[11,"default","ml_rs::classification","",6,[[],["majorityclassifier",3]]],[11,"default","ml_rs::regression::linear","",8,[[],["linearregression",3]]],[11,"fmt","ml_rs","",0,[[["formatter",3]],["result",6]]],[11,"fmt","","",0,[[["formatter",3]],["result",6]]]],"p":[[4,"Error"],[3,"IRLSLogisticRegression"],[3,"LogisticRegression"],[8,"Classifier"],[8,"ProbabilityBinaryClassifier"],[3,"TrivialClassifier"],[3,"MajorityClassifier"],[3,"CsvReader"],[3,"LinearRegression"],[8,"Regressor"],[3,"PrincipalComponentAnalysis"],[8,"Transformer"]]}\
}');
addSearchOptions(searchIndex);initSearch(searchIndex);