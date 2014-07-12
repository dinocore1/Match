package org.devsmart.match;


public interface OfflineClassifier {


    public void addTrainingExample(String className, double[] featureVector);
    public void train();
    public String predict(double[] featureVector);
}
