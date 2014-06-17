package org.devsmart.match;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.optim.PointValuePair;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public abstract class Cart {

    private final int mNumFeatures;

    public class TreeNode {
        double valueOfSplit;
        int featureToSplit;
        TreeNode right;
        TreeNode left;
        MultivariateFunction model;
    }

    public Cart(int numFeatures) {
        mNumFeatures = numFeatures;
    }

    public abstract double errorFunction(Collection<PointValuePair> dataSet);

    public TreeNode createTree(Collection<PointValuePair> dataSet) {


        BestSplit bestSplit = chooseBestSplit(dataSet);


        return null;
    }


    Collection<PointValuePair>[] binSplit(Collection<PointValuePair> dataSet, final int featureIndex, final double value){

        Collection<PointValuePair>[] retval = new Collection[2];

        retval[0] = new ArrayList<PointValuePair>(Collections2.filter(dataSet, new Predicate<PointValuePair>() {
            @Override
            public boolean apply(PointValuePair input) {
                return input.getFirst()[featureIndex] > value;
            }
        }));

        retval[1] = new ArrayList<PointValuePair>(Collections2.filter(dataSet, new Predicate<PointValuePair>() {
            @Override
            public boolean apply(PointValuePair input) {
                return input.getFirst()[featureIndex] <= value;
            }
        }));

        return retval;
    }

    Set<Double> valueSet(Collection<PointValuePair> dataSet, final int featureIndex) {
        return new HashSet<Double>(Collections2.transform(dataSet, new Function<PointValuePair, Double>() {
            @Override
            public Double apply(PointValuePair input) {
                return input.getFirst()[featureIndex];
            }
        }));
    }


    static class BestSplit {
        double bestError;
        double bestValue;
        int bestFeature;
    }

    BestSplit chooseBestSplit(Collection<PointValuePair> dataSet) {

        BestSplit retval = new BestSplit();

        retval.bestError = Double.POSITIVE_INFINITY;
        retval.bestValue = Double.NaN;
        retval.bestFeature = -1;

        for(int featIndex=0;featIndex<mNumFeatures;featIndex++){
            for(double splitValue : valueSet(dataSet, featIndex)){

                Collection<PointValuePair>[] pair = binSplit(dataSet, featIndex, splitValue);

                double error = errorFunction(pair[0]) + errorFunction(pair[1]);
                if(error < retval.bestError){
                    retval.bestError = error;
                    retval.bestFeature = featIndex;
                    retval.bestValue = splitValue;

                }
            }
        }

        return retval;

    }


}
