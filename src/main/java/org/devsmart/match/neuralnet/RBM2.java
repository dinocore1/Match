package org.devsmart.match.neuralnet;


import com.amd.aparapi.Kernel;

public class RBM2 {

    /**
     * Array layout:
     *
     * weight_i_j = (i: visible, j: hidden) numVisible*j+i
     * hiddenbias_j = numVisible*numHidden + j
     * visiblebias_i = numVisible*numHidden + numHidden + i
     */
    final float[] weights;
    final float[] visible;
    final float[] hidden;
    public final int numVisible;
    public final int numHidden;

    private UpKernel upKernel = new UpKernel();
    private DownKernel downKernel = new DownKernel();

    public RBM2(int numVisible, int numHidden) {
        this.numVisible = numVisible;
        this.numHidden = numHidden;
        this.visible = new float[numVisible];
        this.hidden = new float[numHidden];
        this.weights = new float[numVisible*numHidden+numVisible+numHidden];

        upKernel.setExplicit(true);
        downKernel.setExplicit(true);
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

    class UpKernel extends Kernel {

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

    class DownKernel extends Kernel {

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
