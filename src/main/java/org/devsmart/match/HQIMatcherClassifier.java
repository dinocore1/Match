package org.devsmart.match;


import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import java.util.LinkedList;

public class HQIMatcherClassifier implements OfflineClassifier {

    private class TrainingExample{
        String className;
        double[] featureVector;
    }

    private LinkedList<TrainingExample> mExamples = new LinkedList<TrainingExample>();

    @Override
    public void addTrainingExample(String className, double[] featureVector) {
        TrainingExample example = new TrainingExample();
        example.className = className;
        example.featureVector = featureVector;
        mExamples.add(example);
    }

    @Override
    public void train() {
        //no opp
    }

    @Override
    public String predict(double[] featureVector) {
        PearsonsCorrelation correlation = new PearsonsCorrelation();

        String bestClassName = null;
        double bestMatchScore = Double.NaN;
        for(TrainingExample example : mExamples){
            double hqi = correlation.correlation(featureVector, example.featureVector);
            if(Double.isNaN(bestMatchScore) || hqi > bestMatchScore){
                bestClassName = example.className;
                bestMatchScore = hqi;
            }
        }

        return bestClassName;
    }
}
