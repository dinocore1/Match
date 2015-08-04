package org.devsmart.match;


import org.apache.commons.math3.linear.*;

import java.util.Arrays;

public class PLSR {


    private final RealMatrix B0;
    private final RealMatrix B;

    public PLSR(RealMatrix X, double[] y, int l) {
        RealMatrix w = X.transpose().multiply(MatrixUtils.createColumnRealMatrix(y));

        final double norm = w.getNorm();
        w.walkInRowOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value) {
                return value / norm;
            }
        });

        RealMatrix t = X.multiply(w);

        RealMatrix W = MatrixUtils.createRealMatrix(w.getRowDimension(), l+1);
        W.setColumn(0, w.getColumn(0));

        RealMatrix P = MatrixUtils.createRealMatrix(w.getRowDimension(), l+1);

        double[] q = new double[l+1];

        for(int k=0;k<l+1;k++) {
            final double t_k = t.transpose().multiply(t).getEntry(0, 0);

            t.walkInRowOrder(new DefaultRealMatrixChangingVisitor() {
                @Override
                public double visit(int row, int column, double value) {
                    return value / t_k;
                }
            });

            RealMatrix p_k = X.transpose().multiply(t);
            P.setColumn(k, p_k.getColumn(0));

            q[k] = MatrixUtils.createRowRealMatrix(y).multiply(t).getEntry(0,0);
            if(q[k] == 0) {
                l = k;
                break;
            }
            if(k < l) {
                X = X.subtract(t.multiply(p_k.transpose()).scalarMultiply(t_k));
                w = X.transpose().multiply(MatrixUtils.createColumnRealMatrix(y));
                W.setColumn(k+1, w.getColumn(0));
                t = X.multiply(w);
            }
        }


        B = W.multiply( MatrixUtils.inverse(P.transpose().multiply(W)) ).multiply(MatrixUtils.createColumnRealMatrix(q));
        double[] q0 = new double[B.getColumnDimension()];
        Arrays.fill(q0, q[0]);
        B0 = MatrixUtils.createColumnRealMatrix(q0).subtract(P.getColumnMatrix(0).transpose().multiply(B));

        System.out.println("done");


    }

    public double value(double[] x) {
        RealMatrix y = MatrixUtils.createRowRealMatrix(x).multiply(B);
        return y.getEntry(0, 0) + B0.getEntry(0, 0);
    }
}
