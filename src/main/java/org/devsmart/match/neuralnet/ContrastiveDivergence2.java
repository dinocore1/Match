package org.devsmart.match.neuralnet;


import com.amd.aparapi.Kernel;

import java.util.Random;

public class ContrastiveDivergence2 {

    private final RBM2 rbm;

    private float[] positiveVisible;
    private float[] negitiveVisible;

    private float[] positiveHidden;
    private float[] negitiveHidden;
    private float[] gradient;
    private final CDKernel kernel = new CDKernel();

    public ContrastiveDivergence2(RBM2 rbm){
        this.rbm = rbm;
        positiveVisible = new float[rbm.numVisible];
        negitiveVisible = new float[rbm.numVisible];
        positiveHidden = new float[rbm.numHidden];
        negitiveHidden = new float[rbm.numHidden];

        gradient = new float[(rbm.numHidden+1)*(rbm.numVisible+1)];
    }

    private static void sampleVector(float[] v, Random r) {
        for(int i=0;i<v.length;i++){
            v[i] = r.nextDouble() < v[i] ? 1.0f : 0.0f;
        }
    }

    public float[] doIt(float[] visibleInput, int numGibbsSamples, Random r) {

        System.arraycopy(visibleInput,0, positiveVisible, 0, rbm.numVisible);
        float[] values = visibleInput;

        for(int i=0;i<numGibbsSamples;i++){
            values = rbm.activateHidden(values);
            sampleVector(values, r);

            if(i==0) {
                System.arraycopy(values, 0, positiveHidden, 0, rbm.numHidden);
            }

            values = rbm.activateVisible(values);
        }

        System.arraycopy(values, 0, negitiveVisible,0, rbm.numVisible);

        values = rbm.activateHidden(values);
        sampleVector(values, r);
        System.arraycopy(values, 0, negitiveHidden, 0, rbm.numHidden);


        kernel.execute((rbm.numVisible+1)*(rbm.numHidden+1));

        return gradient;
    }

    class CDKernel extends Kernel {

        @Override
        public void run() {
            final int gid = getGlobalId();

            if(gid < rbm.numVisible*rbm.numHidden) {
                gradient[gid] = positiveVisible[gid/rbm.numVisible]*positiveHidden[gid%rbm.numHidden] - negitiveVisible[gid/rbm.numVisible]*negitiveHidden[gid%rbm.numHidden];
            } else if(gid < rbm.numVisible*rbm.numHidden+rbm.numHidden) {
                gradient[gid] = positiveHidden[gid - rbm.numVisible*rbm.numHidden] - negitiveHidden[gid - rbm.numVisible*rbm.numHidden];
            } else {
                gradient[gid] = positiveVisible[gid - rbm.numVisible*rbm.numHidden+rbm.numHidden] - negitiveVisible[gid - rbm.numVisible*rbm.numHidden+rbm.numHidden];
            }
        }
    }

}
