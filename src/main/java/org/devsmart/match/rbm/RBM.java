package org.devsmart.match.rbm;


import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

import java.util.Random;

public class RBM {

    private static final Sigmoid sigmoid = new Sigmoid();

    public final int numVisible;
    public final int numHidden;
    public final boolean isGaussianVisible;

    RealMatrix W;
    //visible bias
    RealVector a;

    //hidden bias
    RealVector b;

    Random r;


    public RBM(int numVisible, int numHidden, boolean isGaussianVisible){
        this(numVisible, numHidden, isGaussianVisible, new Random());
    }

    public RBM(int numVisible, int numHidden, boolean isGaussianVisible, Random r) {
        this.numVisible = numVisible;
        this.numHidden = numHidden;
        this.isGaussianVisible = isGaussianVisible;
        this.r = r;
        this.W = new Array2DRowRealMatrix(numVisible, numHidden);
        this.a = new ArrayRealVector(numVisible);
        this.b = new ArrayRealVector(numHidden);
    }


    //i = visible
    //j = hidden
    public RealVector activateHidden(RealVector visible) {
        RealVector hidden = new ArrayRealVector(numHidden);
        for(int j=0;j<numHidden;j++){
            double sum = W.getColumnVector(j).dotProduct(visible);
            sum += b.getEntry(j);
            double prob = sigmoid.value(sum);
            hidden.setEntry(j, r.nextDouble() < prob ? 1.0 : 0.0);
        }
        return hidden;
    }

    public RealVector activateVisible(RealVector hidden) {
        RealVector visible = new ArrayRealVector(numVisible);
        for(int i=0;i<numVisible;i++){
            double sum = W.getRowVector(i).dotProduct(hidden);
            sum += a.getEntry(i);
            if(isGaussianVisible) {
                visible.setEntry(i, sum);
            } else {
                visible.setEntry(i, sigmoid.value(sum));
            }
        }
        return visible;
    }

    public double freeEnergy(RealVector visible) {
        double sum = 0;
        for(int j=0;j<numHidden;j++){
            sum += FastMath.log(1 + FastMath.exp(b.getEntry(j) + W.getColumnVector(j).dotProduct(visible)));
        }

        return -visible.dotProduct(a) - sum;
    }

}
