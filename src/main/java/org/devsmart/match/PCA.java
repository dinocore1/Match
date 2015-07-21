package org.devsmart.match;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.StatUtils;

public class PCA {

    private RealMatrix mMeans;
    private RealMatrix mTransform;

    public PCA() {

    }

    public PCA(RealMatrix transform, double[] means) {
        mTransform = transform;
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
        mTransform = decomposition.getV();
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

        RealMatrix retval = dataMatrix.multiply(mTransform);

        return retval;

    }


}
