package org.devsmart.match;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.MathUtils;

public class PCA {

    private RealMatrix mMeans;
    private RealMatrix mEigenVectors;
    private double[] mComponentProportions;

    public PCA() {

    }

    public PCA(RealMatrix transform, double[] means) {
        mEigenVectors = transform;
        mMeans = MatrixUtils.createRowRealMatrix(means);
    }

    /**
     * Compute the linear transfrom from a set of observations dataset.
     * @param data the observation data
     * @param rowOrder true if the first dimension of the data array contains the rows
     *                 (observations). If the rowOrder is set to false the first dimension
     *                 of the data array refers to the columns (variables).
     * @param isNormalized true if the data is already mean centered (the columns have a mean of 0).
     */
    public void computeTransform(double[][] data, boolean rowOrder, boolean isNormalized) {

        RealMatrix observations = MatrixUtils.createRealMatrix(data);
        if(!rowOrder) {
            observations = observations.transpose();
        }

        final int numColumns = observations.getColumnDimension();
        if(!isNormalized) {
            double[] means = new double[numColumns];
            for (int i = 0; i < numColumns; i++) {
                means[i] = StatUtils.mean(observations.getColumn(i));
                observations.setColumnVector(i,
                        observations.getColumnVector(i).mapSubtractToSelf(means[i]));
            }
            mMeans = MatrixUtils.createRowRealMatrix(means);
        } else {
            mMeans = MatrixUtils.createRowRealMatrix(new double[numColumns]);
        }


        SingularValueDecomposition decomposition = new SingularValueDecomposition(observations);
        double[] eigenValues = decomposition.getSingularValues();
        for(int i=0;i<eigenValues.length;i++){
            eigenValues[i] = eigenValues[i] * eigenValues[i];
        }
        mComponentProportions = new double[eigenValues.length];
        mEigenVectors = decomposition.getV();

        double sum = StatUtils.sum(eigenValues);


        for(int i=0;i<eigenValues.length;i++) {
            mComponentProportions[i] = eigenValues[i] / sum;
        }

    }

    /**
     * Component proportions are the percent of variance in the data set
     * that is explained by the ith principal component. A number 0 - 1.0
     * @return
     */
    public double[] getComponentProportions() {
        return mComponentProportions;
    }

    /**
     * Perform the PCA transform on a new set data entry and return its principal components.
     * @param data
     * @param isNormalized true if the data is already mean centered (the columns have a mean of 0).
     * @return
     */
    public RealMatrix getPrincipalComponents(double[] data, boolean isNormalized) {

        RealMatrix dataMatrix = MatrixUtils.createRowRealMatrix(data);

        if(!isNormalized) {
            dataMatrix = dataMatrix.subtract(mMeans);
        }

        RealMatrix retval = dataMatrix.multiply(mEigenVectors);

        return retval;

    }


}
