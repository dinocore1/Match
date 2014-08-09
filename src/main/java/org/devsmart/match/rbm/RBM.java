package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;
import org.devsmart.match.rbm.nuron.Nuron;

import java.util.Random;

public class RBM {

    final Nuron[] visible;
    final Nuron[] hidden;

    //weights
    RealMatrix W;

    //visible bias
    RealVector a;

    //hidden bias
    RealVector b;

    public RBM(Nuron[] visible, Nuron[] hidden) {
        this.visible = visible;
        this.hidden = hidden;
        this.W = new Array2DRowRealMatrix(visible.length, hidden.length);
        this.a = new ArrayRealVector(visible.length);
        this.b = new ArrayRealVector(hidden.length);
    }

    public RealVector getHiddenInput(RealVector visible) {
        RealVector retval = new ArrayRealVector(hidden.length);
        for(int j=0;j<hidden.length;j++){
            double sum = W.getColumnVector(j).dotProduct(visible);
            sum += b.getEntry(j);
            double prob = hidden[j].activate(sum);
            retval.setEntry(j, prob);
        }
        return retval;
    }

    public RealVector getVisibleInput(RealVector hidden) {
        RealVector retval = new ArrayRealVector(visible.length);
        for(int i=0;i<visible.length;i++){
            double sum = W.getRowVector(i).dotProduct(hidden);
            sum += a.getEntry(i);
            double prob = visible[i].activate(sum);
            retval.setEntry(i, prob);
        }
        return retval;
    }

    public RealVector activateHidden(RealVector visible, Random r) {
        RealVector retval = new ArrayRealVector(hidden.length);
        for(int j=0;j<hidden.length;j++){
            double sum = W.getColumnVector(j).dotProduct(visible);
            sum += b.getEntry(j);
            double prob = hidden[j].activate(sum);
            retval.setEntry(j, r.nextDouble() < prob ? 1.0 : 0.0);
        }
        return retval;
    }

    public RealVector activateVisible(RealVector hidden, Random r) {
        RealVector retval = new ArrayRealVector(visible.length);
        for(int i=0;i<visible.length;i++){
            double sum = W.getRowVector(i).dotProduct(hidden);
            sum += a.getEntry(i);
            double prob = visible[i].activate(sum);
            retval.setEntry(i, r.nextDouble() < prob ? 1.0 : 0.0);
        }
        return retval;
    }

    public double freeEnergy(RealVector visible) {
        double sum = 0;
        for(int j=0;j<hidden.length;j++){
            sum += FastMath.log(1 + FastMath.exp(b.getEntry(j) + W.getColumnVector(j).dotProduct(visible)));
        }

        return -visible.dotProduct(a) - sum;
    }

}
