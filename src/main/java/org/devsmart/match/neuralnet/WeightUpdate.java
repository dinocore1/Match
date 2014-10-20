package org.devsmart.match.neuralnet;


import com.amd.aparapi.Kernel;

public class WeightUpdate {


    private final RBM2 rbm;
    private final float[] lastWeights;
    private float learningrate = 0.1f;
    private float momentum = 0.1f;


    public WeightUpdate(RBM2 rbm, float learningRate) {
        this.rbm = rbm;
        this.lastWeights = new float[rbm.numVisible*rbm.numHidden+rbm.numVisible+rbm.numHidden];
    }

    public synchronized void setLearningrate(float learningrate) {
        this.learningrate = learningrate;
    }

    public synchronized void setMomentum(float momentum) {
        this.momentum = momentum;
    }

    public synchronized float[] update(float[] gradientInput) {
        assert gradientInput.length == lastWeights.length;

        final float[] weights = rbm.getWeights();

        for(int i=0;i<lastWeights.length;i++){
            weights[i] += learningrate * (gradientInput[i] + momentum*lastWeights[i]);
        }

        rbm.setWeights(weights);
        return weights;
    }

}
