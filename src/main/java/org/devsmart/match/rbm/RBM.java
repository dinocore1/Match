package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;
import org.devsmart.match.rbm.nuron.Neuron;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Random;

public class RBM {

    final Neuron[] visible;
    final Neuron[] hidden;

    //weights
    RealMatrix W;

    //visible bias
    RealVector a;

    //hidden bias
    RealVector b;

    protected RBM(int visible, int hidden) {
        this.visible = new Neuron[visible];
        this.hidden = new Neuron[hidden];
    }

    public RBM(Neuron[] visible, Neuron[] hidden) {
        this.visible = visible;
        this.hidden = hidden;
        this.W = new Array2DRowRealMatrix(visible.length, hidden.length);
        this.a = new ArrayRealVector(visible.length);
        this.b = new ArrayRealVector(hidden.length);
    }

    public void save(OutputStream out) throws IOException {
        DataOutputStream dout = new DataOutputStream(out);
        dout.writeInt(W.getRowDimension());
        dout.writeInt(W.getColumnDimension());
        for(int i=0;i<W.getRowDimension();i++){
            for(int j=0;j<W.getColumnDimension();j++){
                double value = W.getEntry(i, j);
                dout.writeDouble(value);
            }
        }
        for(int i=0;i<a.getDimension();i++){
            dout.writeDouble(a.getEntry(i));
        }
        for(int i=0;i<b.getDimension();i++){
            dout.writeDouble(b.getEntry(i));
        }
        dout.flush();
    }

    public static RBM load(DataInputStream din) throws IOException {
        final int visible = din.readInt();
        final int hidden = din.readInt();

        RBM retval = new RBM(visible, hidden);
        retval.W = new Array2DRowRealMatrix(visible, hidden);
        for(int i=0;i<visible;i++){
            for(int j=0;j<hidden;j++){
                double value = din.readDouble();
                retval.W.setEntry(i, j, value);
            }
        }

        retval.a = new ArrayRealVector(visible);
        for(int i=0;i<visible;i++){
            double value = din.readDouble();
            retval.a.setEntry(i, value);
        }

        retval.b = new ArrayRealVector(hidden);
        for(int i=0;i<hidden;i++){
            double value = din.readDouble();
            retval.b.setEntry(i, value);
        }


        return retval;
    }

    public double[] getHiddenInput(double[] visible) {
        double[] retval = new double[hidden.length];
        for(int j=0;j<hidden.length;j++){
            double sum = W.getColumnVector(j).dotProduct(new ArrayRealVector(visible));
            sum += b.getEntry(j);
            retval[j] = sum;
        }
        return retval;
    }

    public double[] getVisibleInput(double[] hidden) {
        double[] retval = new double[visible.length];
        for(int i=0;i<visible.length;i++){
            double sum = W.getRowVector(i).dotProduct(new ArrayRealVector(hidden));
            sum += a.getEntry(i);
            retval[i] = sum;
        }
        return retval;
    }

    public double[] activateHidden(double[] visible) {
        double[] retval = new double[hidden.length];
        for(int j=0;j<hidden.length;j++){
            double sum = W.getColumnVector(j).dotProduct(new ArrayRealVector(visible));
            sum += b.getEntry(j);
            retval[j] = hidden[j].activate(sum);
        }
        return retval;
    }

    public double[] sampleHidden(double[] visible, Random r) {
        double[] retval = new double[hidden.length];
        for(int j=0;j<hidden.length;j++){
            double sum = W.getColumnVector(j).dotProduct(new ArrayRealVector(visible));
            sum += b.getEntry(j);
            double prob = hidden[j].activate(sum);
            retval[j] = r.nextDouble() < prob ? 1.0 : 0.0;
        }
        return retval;
    }

    public double[] activateVisible(double[] hidden) {
        double[] retval = new double[visible.length];
        for(int i=0;i<visible.length;i++){
            double sum = W.getRowVector(i).dotProduct(new ArrayRealVector(hidden));
            sum += a.getEntry(i);
            retval[i] = visible[i].activate(sum);
        }
        return retval;
    }

    public double[] sampleVisible(double[] hidden, Random r) {
        double[] retval = new double[visible.length];
        for(int i=0;i<visible.length;i++){
            double sum = W.getRowVector(i).dotProduct(new ArrayRealVector(hidden));
            sum += a.getEntry(i);
            double prob = visible[i].activate(sum);
            retval[i] = r.nextDouble() < prob ? 1.0 : 0.0;
        }
        return retval;
    }

    public double freeEnergy(double[] visible) {
        ArrayRealVector visibleVector = new ArrayRealVector(visible);
        double sum = 0;
        for(int j=0;j<hidden.length;j++){
            sum += FastMath.log(1 + FastMath.exp(b.getEntry(j) + W.getColumnVector(j).dotProduct(visibleVector)));
        }

        return -visibleVector.dotProduct(a) - sum;
    }

}
