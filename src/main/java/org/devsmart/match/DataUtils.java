package org.devsmart.match;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

public class DataUtils {

    /**
     * Standards the data observations to have mean of 0 and a standard deviation of 1.
     * Returns a Matrix in which rows are observations and columns are variables.
     *
     * @param data
     * @param rowOrder true if the first dimension of the data array contains the rows
     *                 (observations). If the rowOrder is set to false the first dimension
     *                 of the data array refers to the columns (variables).
     * @return
     */

    public static RealMatrix standardize(double[][] data, final boolean rowOrder) {
        RealMatrix dataMatrix = MatrixUtils.createRealMatrix(data);
        if(!rowOrder) {
            dataMatrix.transpose();
        }

        for(int j=0;j<dataMatrix.getColumnDimension();j++) {
            dataMatrix.setColumn(j, StatUtils.normalize(dataMatrix.getColumn(j)));
        }

        return dataMatrix;
    }

    /**
     * Standards the data observations to have mean of 0.
     * Returns a Matrix in which rows are observations and columns are variables.
     *
     * @param data
     * @param rowOrder true if the first dimension of the data array contains the rows
     *                 (observations). If the rowOrder is set to false the first dimension
     *                 of the data array refers to the columns (variables).
     * @return
     */
    public static RealMatrix centerMean(double[][] data, final boolean rowOrder) {
        RealMatrix dataMatrix = MatrixUtils.createRealMatrix(data);
        if(!rowOrder) {
            dataMatrix.transpose();
        }

        for(int j=0;j<dataMatrix.getColumnDimension();j++) {
            double[] column = dataMatrix.getColumn(j);
            zeroMean(column);
            dataMatrix.setColumn(j, column);
        }

        return dataMatrix;
    }

    public static void zeroMean(double[] data) {
        double dataMean = StatUtils.mean(data);
        for(int i=0;i<data.length;i++){
            data[i] = data[i] - dataMean;
        }
    }

}
