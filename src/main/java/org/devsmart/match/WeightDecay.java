package org.devsmart.match;


import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public interface WeightDecay {

    RealMatrix getWeightPenalty(RealMatrix W, double coefficient);

    WeightDecay L1 = new WeightDecay() {

        @Override
        public RealMatrix getWeightPenalty(RealMatrix W, final double coefficient) {
            Array2DRowRealMatrix retval = new Array2DRowRealMatrix(W.getRowDimension(), W.getColumnDimension());

            for(int i=0;i<retval.getRowDimension();i++){
                for(int j=0;j<retval.getColumnDimension();j++){
                    retval.setEntry(i, j, coefficient * Math.abs(W.getEntry(i, j)));
                }
            }


            return retval;
        }
    };
}
