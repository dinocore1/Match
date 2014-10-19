package org.devsmart.match.neuralnet;


import com.amd.aparapi.Kernel;

import java.util.Random;

public class ContrastiveDivergence2 {

    private static final int MINIBATCHCOEFF = 0;

    private final RBM2 rbm;

    private float[] positiveVisible;
    private float[] negitiveVisible;

    private float[] positiveHidden;
    private float[] negitiveHidden;
    private float[] gradient;
    private float[] params = new float[1];
    private final CDKernel kernel = new CDKernel();

    public ContrastiveDivergence2(RBM2 rbm){
        this.rbm = rbm;
        positiveVisible = new float[rbm.numVisible];
        negitiveVisible = new float[rbm.numVisible];
        positiveHidden = new float[rbm.numHidden];
        negitiveHidden = new float[rbm.numHidden];

        gradient = new float[rbm.numVisible*rbm.numHidden+rbm.numVisible+rbm.numHidden];

        params[MINIBATCHCOEFF] = 1.0f;
        kernel.setExplicit(true);
    }

    private static void sampleVector(float[] v, Random r) {
        for(int i=0;i<v.length;i++){
            v[i] = r.nextDouble() < v[i] ? 1.0f : 0.0f;
        }
    }

    public void resetGradient() {
        for(int i=0;i<gradient.length;i++){
            gradient[i] = 0;
        }
        kernel.put(gradient);
    }

    public void setMinibatchSize(int size) {
        params[MINIBATCHCOEFF] = 1f / (float)size;
        kernel.put(params);
    }

    public void doIt(double[] visibleInput, int numGibbsSamples, Random r) {
        assert visibleInput.length == rbm.numVisible;

        for(int i=0;i<rbm.numVisible;i++) {
            positiveVisible[i] = (float)visibleInput[i];
        }

        doIt(numGibbsSamples, r);
    }

    public void doIt(float[] visibleInput, int numGibbsSamples, Random r) {
        assert visibleInput.length == rbm.numVisible;

        System.arraycopy(visibleInput,0, positiveVisible, 0, rbm.numVisible);

        doIt(numGibbsSamples, r);
    }

    public void doIt(int numGibbsSamples, Random r) {

        float[] values = positiveVisible;

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


        kernel.put(positiveVisible);
        kernel.put(positiveHidden);
        kernel.put(negitiveVisible);
        kernel.put(negitiveHidden);

        kernel.execute(rbm.numVisible*rbm.numHidden+rbm.numVisible+rbm.numHidden);
    }

    public float[] getGradient() {
        kernel.get(gradient);
        return gradient;
    }

    class CDKernel extends Kernel {

        @Override
        public void run() {
            final int gid = getGlobalId();

            if(gid < rbm.numVisible*rbm.numHidden) {
                gradient[gid] += params[MINIBATCHCOEFF] * (positiveVisible[gid/rbm.numVisible]*positiveHidden[gid%rbm.numHidden] - negitiveVisible[gid/rbm.numVisible]*negitiveHidden[gid%rbm.numHidden]);
            } else if(gid < (rbm.numVisible*rbm.numHidden+rbm.numHidden)) {
                gradient[gid] += params[MINIBATCHCOEFF] * ( positiveHidden[gid - rbm.numVisible*rbm.numHidden] - negitiveHidden[gid - rbm.numVisible*rbm.numHidden] );
            } else {
                gradient[gid] += params[MINIBATCHCOEFF] * ( positiveVisible[gid - (rbm.numVisible*rbm.numHidden+rbm.numHidden)] - negitiveVisible[gid - (rbm.numVisible*rbm.numHidden+rbm.numHidden)] );
            }

        }
    }

}
