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

    public static class TreeNode {
        double valueOfSplit;
        int featureToSplit;
        TreeNode right;
        TreeNode left;
        MultivariateFunction model;

        boolean isLeaf() {
            return model != null;
        }
    }

    public Cart(int numFeatures) {
        mNumFeatures = numFeatures;
    }

    protected abstract MultivariateFunction createModel(Collection<PointValuePair> dataSet);
    protected abstract double errorFunction(Collection<PointValuePair> dataSet);
    protected abstract boolean shouldContinue(BestSplit split);

    public TreeNode createTree(Collection<PointValuePair> dataSet) {

        TreeNode retval = null;

        BestSplit bestSplit = chooseBestSplit(dataSet);
        if(shouldContinue(bestSplit)){
            retval = new TreeNode();
            retval.featureToSplit = bestSplit.bestFeature;
            retval.valueOfSplit = bestSplit.bestValue;
            retval.left = createTree(bestSplit.left);
            retval.right = createTree(bestSplit.right);

        } else {
            retval = new TreeNode();
            retval.model = createModel(dataSet);
        }


        return retval;
    }

    public static double value(double[] point, TreeNode tree) {
        if(tree.isLeaf()){
            return tree.model.value(point);
        } else {
            double value = point[tree.featureToSplit];
            if(value > tree.valueOfSplit) {
                return value(point, tree.right);
            } else {
                return value(point, tree.left);
            }
        }
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
        Collection<PointValuePair> right;
        Collection<PointValuePair> left;
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
                    retval.right = pair[0];
                    retval.left = pair[1];

                }
            }
        }

        return retval;

    }


}
