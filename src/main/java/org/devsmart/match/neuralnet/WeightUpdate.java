package org.devsmart.match.neuralnet;


import com.amd.aparapi.Kernel;

public class WeightUpdate {

    private static final int LEARNINGRATE = 0;
    private static final int MOMENTUM = 1;

    public final int size;
    private final float[] weights;
    private final float[] gradient;

    private final float[] params = new float[2];
    private final Kernel kernel = new UpdateKernel();


    public WeightUpdate(int size, float learningRate) {
        this.size = size;
        this.weights = new float[size];
        this.gradient = new float[size];

        params[LEARNINGRATE] = learningRate;
        params[MOMENTUM] = 0.1f;
    }

    public synchronized void setLearningrate(float learningrate) {
        params[LEARNINGRATE] = learningrate;
        kernel.put(params);
    }

    public synchronized void setMomentum(float momentum) {
        params[MOMENTUM] = momentum;
        kernel.put(params);
    }

    public synchronized void update(float[] gradientInput) {
        assert gradientInput.length == gradient.length;
        System.arraycopy(gradientInput, 0, gradient, 0, gradient.length);


        kernel.put(gradient);
        kernel.execute(size);
        kernel.get(weights);
    }

    class UpdateKernel extends Kernel {

        @Override
        public void run() {
            final int gid = getGlobalId();

            weights[gid] += params[LEARNINGRATE] * (gradient[gid] + params[MOMENTUM]*weights[gid]);

        }
    }
}
