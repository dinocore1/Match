package org.devsmart.match.neuralnet;


import org.devsmart.match.rbm.nuron.LinearNuron;
import org.devsmart.match.rbm.nuron.Neuron;
import org.devsmart.match.rbm.nuron.SigmoidNuron;
import org.devsmart.match.rbm.nuron.SoftPlus;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

public class FlipperTest {

    static final int haystackSize = 5;

    private static class HaystackTest {

        double[] input;
        final double needle;

        public HaystackTest(Random r) {
            input = new double[haystackSize+1];
            needle = r.nextDouble();

            for(int i=1;i<haystackSize;i++){
                input[i] = r.nextDouble();
            }

            int q = r.nextInt(haystackSize);
            input[0] = q;
            input[q+1] = needle;
        }
    }


    @Test
    public void testFlipper() {

        final Random r = new Random(1);


        MiniBatchCreator miniBatchCreator = new MiniBatchCreator() {
            @Override
            public Collection<TraningData> createMiniBatch() {
                ArrayList<TraningData> retval = new ArrayList<TraningData>();
                for(int i=0;i<1;i++){
                    TraningData data = new TraningData();

                    HaystackTest t = new HaystackTest(r);
                    data.intput = t.input;
                    data.output = new double[]{t.needle};

                    retval.add(data);
                }
                return retval;
            }
        };

        FreeFormNNBuilder builder = new FreeFormNNBuilder();

        Neuron[] inputLayer = builder.createLayer(haystackSize+1, LinearNuron.class);

        Neuron[] sigmoidH = builder.createLayer(haystackSize, SigmoidNuron.class);
        builder.addBias(sigmoidH);

        for(int i=0;i<sigmoidH.length;i++){
            builder.addConnection(inputLayer[0], sigmoidH[i], r.nextGaussian());
        }

        Neuron[] rlinearH = builder.createLayer(haystackSize, SoftPlus.class);
        builder.addBias(rlinearH);
        builder.fullyConnect(sigmoidH, rlinearH);

        for(int i=0;i<haystackSize;i++){
            builder.addConnection(inputLayer[i+1], rlinearH[i], 1 + 0.1*r.nextGaussian());
        }

        Neuron[] outputLayer = builder.createLayer(1, LinearNuron.class);
        for(int i=0;i<haystackSize;i++){
            builder.addConnection(rlinearH[i], outputLayer[0], 1);
        }

        NeuralNet nn = builder.build();
        Backpropagation backpropagation = new Backpropagation();
        backpropagation.neuralNet = nn;
        backpropagation.learningRate = 0.1;
        backpropagation.momentum = 0.2;
        backpropagation.miniBatchCreator = miniBatchCreator;
        backpropagation.train(0.0001, 1000000);


        for(int i=0;i<100;i++){
            HaystackTest t = new HaystackTest(r);

            NeuralNet.Context ctx = nn.setInputs(t.input);
            final double output = ctx.getOutputs()[0];
            System.out.println(String.format("i: %s needle: %g output: %g", Arrays.toString(t.input), t.needle, output));
            assertEquals(t.needle, output, 0.2);
        }

    }
}
