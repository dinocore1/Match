package org.devsmart.match;


import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.Collection;
import java.util.Iterator;

public class BasicCART extends Cart {


    public BasicCART(int numFeatures) {
        super(numFeatures);
    }

    @Override
    protected MultivariateFunction createModel(Collection<PointValuePair> dataSet) {
        Array2DRowRealMatrix matrix = null;
        ArrayRealVector yVector = new ArrayRealVector(dataSet.size());
        Iterator<PointValuePair> it = dataSet.iterator();

        int r = 0;
        while(it.hasNext()){
            PointValuePair sample = it.next();

            if(matrix == null){
                matrix = new Array2DRowRealMatrix(dataSet.size(), sample.getFirst().length);
            }
            matrix.setRow(r, sample.getFirst());
            yVector.setEntry(r, sample.getSecond());

            r++;
        }

        SingularValueDecomposition decomposition = new SingularValueDecomposition(matrix);
        final RealVector paramsVector = decomposition.getSolver().solve(yVector);

        return new MultivariateFunction() {
            @Override
            public double value(double[] point) {
                Array2DRowRealMatrix values = new Array2DRowRealMatrix(1, point.length);
                values.setRow(0, point);

                Array2DRowRealMatrix coefficients = new Array2DRowRealMatrix(point.length, 1);
                coefficients.setColumn(0, paramsVector.toArray());

                Array2DRowRealMatrix result = values.multiply(coefficients);
                return result.getEntry(0, 0);
            }
        };
    }

    @Override
    protected double errorFunction(Collection<PointValuePair> dataSet) {
        SummaryStatistics error = new SummaryStatistics();

        MultivariateFunction modelFunction = createModel(dataSet);
        double[] yValues = new double[dataSet.size()];
        double[] xValues = new double[dataSet.size()];

        Iterator<PointValuePair> it = dataSet.iterator();
        int i = 0;
        while(it.hasNext()){
            PointValuePair sample = it.next();

            yValues[i] = sample.getSecond();
            xValues[i] = modelFunction.value(sample.getFirst());

            error.addValue(yValues[i] - xValues[i]);
            i++;
        }

        double RMSE = Math.sqrt( error.getSumsq() / error.getN() );
        return RMSE;
    }

    @Override
    protected boolean shouldContinue(BestSplit split) {
        if(split == null || split.left == null || split.right == null){
            System.out.println("something was null");
            return false;
        } else {
            return split.bestError > 0.1 && (split.left.size() > 1 && split.right.size() > 1);
        }
    }
}
