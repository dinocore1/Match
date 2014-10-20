package org.devsmart.match.neuralnet;


import com.amd.aparapi.Kernel;

import java.util.Random;

public class ContrastiveDivergence2 {

    private final RBM2 rbm;
    private final CDKernel kernel;

    public ContrastiveDivergence2(RBM2 rbm){
        this.rbm = rbm;
        this.kernel = new CDKernel(rbm);
        kernel.setExplicit(true);
    }

    private static void sampleVector(float[] v, Random r) {
        for(int i=0;i<v.length;i++){
            v[i] = r.nextDouble() < v[i] ? 1.0f : 0.0f;
        }
    }

    public void resetGradient() {
        for(int i=0;i<kernel.gradient.length;i++){
            kernel.gradient[i] = 0;
        }
        kernel.put(kernel.gradient);
    }

    public void doIt(double[] visibleInput, int numGibbsSamples, Random r) {
        assert visibleInput.length == rbm.numVisible;

        for(int i=0;i<rbm.numVisible;i++) {
            kernel.positiveVisible[i] = (float)visibleInput[i];
        }

        doIt(numGibbsSamples, r);
    }

    public void doIt(float[] visibleInput, int numGibbsSamples, Random r) {
        assert visibleInput.length == rbm.numVisible;

        System.arraycopy(visibleInput,0, kernel.positiveVisible, 0, rbm.numVisible);

        doIt(numGibbsSamples, r);
    }

    public void doIt(int numGibbsSamples, Random r) {

        float[] values = kernel.positiveVisible;

        for(int i=0;i<numGibbsSamples;i++){
            values = rbm.activateHidden(values);
            sampleVector(values, r);

            if(i==0) {
                System.arraycopy(values, 0, kernel.positiveHidden, 0, rbm.numHidden);
            }

            values = rbm.activateVisible(values);
        }

        System.arraycopy(values, 0, kernel.negitiveVisible,0, rbm.numVisible);

        values = rbm.activateHidden(values);
        sampleVector(values, r);
        System.arraycopy(values, 0, kernel.negitiveHidden, 0, rbm.numHidden);


        kernel.put(kernel.positiveVisible);
        kernel.put(kernel.positiveHidden);
        kernel.put(kernel.negitiveVisible);
        kernel.put(kernel.negitiveHidden);
        kernel.put(kernel.gradient);

        kernel.execute(rbm.numVisible*rbm.numHidden+rbm.numVisible+rbm.numHidden);
        kernel.get(kernel.gradient);
    }

    public float[] getGradient() {
        kernel.get(kernel.gradient);
        return kernel.gradient;
    }

    static class CDKernel extends Kernel {

        private final float[] positiveVisible;
        private final float[] negitiveVisible;

        private final float[] positiveHidden;
        private final float[] negitiveHidden;
        private final float[] gradient;

        final int numVisible;
        final int numHidden;

        public CDKernel(RBM2 rbm) {
            numVisible = rbm.numVisible;
            numHidden = rbm.numHidden;

            positiveVisible = new float[rbm.numVisible];
            negitiveVisible = new float[rbm.numVisible];
            positiveHidden = new float[rbm.numHidden];
            negitiveHidden = new float[rbm.numHidden];
            gradient = new float[rbm.numVisible*rbm.numHidden+rbm.numVisible+rbm.numHidden];
        }

        @Override
        public void run() {
            final int gid = getGlobalId();

            if(gid < numVisible*numHidden) {
                gradient[gid] += (positiveVisible[gid%numVisible]*positiveHidden[gid/numVisible] - negitiveVisible[gid%numVisible]*negitiveHidden[gid/numVisible]);
            } else if(gid < (numVisible*numHidden+numHidden)) {
                gradient[gid] += ( positiveHidden[gid - numVisible*numHidden] - negitiveHidden[gid - numVisible*numHidden] );
            } else {
                gradient[gid] += ( positiveVisible[gid - (numVisible*numHidden+numHidden)] - negitiveVisible[gid - (numVisible*numHidden+numHidden)] );
            }

        }
    }

}
