package org.devsmart.match;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.ParametricUnivariateFunction;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.optim.PointValuePair;

import java.util.Collection;

public class Cart {

    public class TreeNode {
        double valueOfSplit;
        int featureToSplit;
        TreeNode right;
        TreeNode left;
        MultivariateFunction model;
    }


    public TreeNode createTree(Collection<PointValuePair> dataSet) {
        if(dataSet.size() <= ){

        }
    }


}
