package org.devsmart.match.neuralnet;


import com.amd.aparapi.Kernel;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class RBM2 {

    Logger logger = LoggerFactory.getLogger(RBM2.class);

    /**
     * Array layout:
     *
     * weight_i_j = (i: visible, j: hidden) numVisible*j+i
     * hiddenbias_j = numVisible*numHidden + j
     * visiblebias_i = numVisible*numHidden + numHidden + i
     */
    private final float[] weights;
    private final float[] visible;
    private final float[] hidden;
    public final int numVisible;
    public final int numHidden;

    private final UpKernel upKernel;
    private final DownKernel downKernel;

    public RBM2(int numVisible, int numHidden) {
        this.numVisible = numVisible;
        this.numHidden = numHidden;
        this.visible = new float[numVisible];
        this.hidden = new float[numHidden];
        this.weights = new float[numVisible*numHidden+numVisible+numHidden];

        upKernel = new UpKernel(weights, visible, hidden);
        upKernel.setExplicit(true);

        downKernel = new DownKernel(weights, visible, hidden);
        downKernel.setExplicit(true);
    }

    public static int getWeight(final int numVisible, final int numHidden, final int i, final int j) {
        return numVisible*j+i;
    }

    public int getWeight(final int i, final int j) {
        return getWeight(numVisible, numHidden, i, j);
    }

    public static int getHiddenBias(final int numVisible, final int numHidden, final int j){
        return numVisible*numHidden+j;
    }

    public int getHiddenBias(int j) {
        return getHiddenBias(numVisible, numHidden, j);
    }

    public static int getVisibleBias(final int numVisible, final int numHidden, final int i){
        return numVisible*numHidden + numHidden + i;
    }

    public int getVisibleBias(int i) {
        return getVisibleBias(numVisible, numHidden, i);
    }

    public synchronized void save(OutputStream out) throws IOException {
        DataOutputStream dout = new DataOutputStream(out);
        dout.writeInt(numVisible);
        dout.writeInt(numHidden);
        for(int i=0;i<numVisible;i++){
            for(int j=0;j<numHidden;j++) {
                dout.writeDouble(weights[getWeight(i, j)]);
            }
        }
        for(int i=0;i<numVisible;i++){
            dout.writeDouble(weights[getVisibleBias(i)]);
        }
        for(int j=0;j<numHidden;j++){
            dout.writeDouble(weights[getHiddenBias(j)]);
        }
        dout.flush();
    }

    public static RBM2 load(InputStream in) throws IOException {
        DataInputStream din = new DataInputStream(in);
        int numVisible = din.readInt();
        int numHidden = din.readInt();
        RBM2 retval = new RBM2(numVisible, numHidden);
        for(int i=0;i<numVisible;i++){
            for(int j=0;j<numHidden;j++) {
                retval.weights[retval.getWeight(i, j)] = (float)din.readDouble();
            }
        }
        for(int i=0;i<numVisible;i++){
            retval.weights[retval.getVisibleBias(i)] = (float)din.readDouble();
        }
        for(int j=0;j<numHidden;j++){
            retval.weights[retval.getHiddenBias(j)] = (float)din.readDouble();
        }
        return retval;
    }

    public synchronized float[] activateHidden(double[] visibleInput) {
        assert visibleInput.length == numVisible;
        for(int i=0;i<numVisible;i++) {
            visible[i] = (float) visibleInput[i];
        }

        return activateHidden();
    }

    public synchronized float[] activateHidden(float[] visibleInput) {
        assert visibleInput.length == numVisible;
        System.arraycopy(visibleInput, 0, visible, 0, numVisible);

        return activateHidden();
    }

    public synchronized float[] activateHidden() {
        upKernel.put(visible);
        upKernel.execute(numHidden);
        upKernel.get(hidden);

        return hidden;
    }

    public synchronized float[] activateVisible(float[] hiddenInput) {
        assert hiddenInput.length == numHidden;
        System.arraycopy(hiddenInput, 0, hidden, 0, numHidden);

        downKernel.put(hidden);
        downKernel.execute(numVisible);
        downKernel.get(visible);

        return visible;
    }

    public synchronized void setWeights(float[] weightsInput) {
        assert weightsInput.length == weights.length;
        System.arraycopy(weightsInput, 0, weights, 0, weights.length);

        upKernel.put(weights);
        downKernel.put(weights);
    }

    public synchronized float[] getWeights() {
        return weights;
    }

    static class UpKernel extends Kernel {

        final float[] weights;
        final float[] visible;
        final float[] hidden;
        private final int numVisible;
        private final int numHidden;

        public UpKernel(float[] weights, float[] visible, float[] hidden) {
            this.weights = weights;
            this.visible = visible;
            this.hidden = hidden;

            this.numVisible = visible.length;
            this.numHidden = hidden.length;
        }



        @Override
        public void run() {

            int gid = getGlobalId();

            float sum = 0;
            for(int i=0;i<numVisible;i++) {
                sum += weights[numVisible*gid + i] * visible[i];
            }

            //add bias
            sum += weights[(numVisible*numHidden) + gid];

            hidden[gid] = 1.0f / (1.0f + exp(-sum));
        }
    }

    static class DownKernel extends Kernel {

        final float[] weights;
        final float[] visible;
        final float[] hidden;
        private final int numVisible;
        private final int numHidden;

        public DownKernel(float[] weights, float[] visible, float[] hidden) {
            this.weights = weights;
            this.visible = visible;
            this.hidden = hidden;

            this.numVisible = visible.length;
            this.numHidden = hidden.length;
        }

        @Override
        public void run() {
            int gid = getGlobalId();

            float sum = 0;
            for(int i=0;i<numHidden;i++) {
                sum += weights[numVisible*i + gid] * hidden[i];
            }

            //add bias
            sum += weights[(numVisible*numHidden) + numHidden + gid];

            visible[gid] = 1.0f / (1.0f + exp(-sum));
        }
    }



}
