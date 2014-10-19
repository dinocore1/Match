package org.devsmart.match.neuralnet;

import com.amd.aparapi.Kernel;


public class SigmoidKernel extends Kernel {


    public final float[] weights;
    public final float[] inputValues;
    public final float[] outputValues;

    public final int numInputs;
    private final int numOutputs;


    public SigmoidKernel(int numInputs, int numOutputs) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;

        weights = new float[(numInputs+1)*numOutputs];
        inputValues = new float[numInputs];
        outputValues = new float[numOutputs];

    }

    @Override
    public void run() {
        int gid = getGlobalId();


        float sum = 0;
        for(int i=0;i<numInputs;i++) {
            sum += weights[(numInputs+1)*gid + i] * inputValues[i];
        }
        //last one is the bias
        sum += weights[(numInputs+1)*gid + numInputs];

        outputValues[gid] = 1.0f / (1.0f + exp(-sum));
    }
}
