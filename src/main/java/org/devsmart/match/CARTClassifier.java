package org.devsmart.match;


import com.google.common.collect.HashMultimap;
import org.apache.commons.math3.optim.PointValuePair;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

public class CARTClassifier implements OfflineClassifier {

    private transient HashMultimap<String, double[]> mTrainingData = HashMultimap.create();
    private transient int mNumFeatures;

    private HashMap<String, Cart.TreeNode> mClassifiers = new HashMap<String, Cart.TreeNode>();



    @Override
    public void addTrainingExample(String className, double[] featureVector) {
        mNumFeatures = featureVector.length;
        mTrainingData.put(className, featureVector);
    }

    @Override
    public void train() {

        for(String className : mTrainingData.keySet()){
            BasicCART cart = new BasicCART(mNumFeatures);


            LinkedList<PointValuePair> pointPairs = new LinkedList<PointValuePair>();
            for(Map.Entry<String, double[]> entry : mTrainingData.entries()){
                double value = className.equals(entry.getKey()) ? 1.0 : 0.0;
                pointPairs.add(new PointValuePair(entry.getValue(), value));
            }
            Cart.TreeNode tree = cart.createTree(pointPairs);
            mClassifiers.put(className, tree);
        }


    }

    @Override
    public String predict(double[] featureVector) {
        String bestClass = null;
        double bestDiff = Double.NaN;

        for(Map.Entry<String, Cart.TreeNode> entry : mClassifiers.entrySet()){
            double match = Cart.value(featureVector, entry.getValue());
            double diff = Math.abs(1.0 - match);
            System.out.println(String.format("%s %s diff %s", entry.getKey(), match, diff));
            if(Double.isNaN(bestDiff) || diff < bestDiff){
                bestClass = entry.getKey();
                bestDiff = diff;
            }
        }

        return bestClass;
    }
}
