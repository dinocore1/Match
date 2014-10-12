package org.devsmart.match.neuralnet;


import org.devsmart.match.rbm.nuron.LinearNuron;
import org.devsmart.match.rbm.nuron.SigmoidNuron;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collection;

public class XORTest {

    static ArrayList<TraningData> truthTable = new ArrayList<TraningData>();

    /**
     * +----------------+
     * | a | b | output |
     * +---+---+--------+
     * | 0 | 0 |  0     |
     * | 0 | 1 |  1     |
     * | 1 | 0 |  1     |
     * | 1 | 1 |  0     |
     * +----------------+
     */
    static {

        TraningData data = new TraningData();
        data.intput = new double[] {0, 0};
        data.output = new double[] {0};
        truthTable.add(data);

        data = new TraningData();
        data.intput = new double[] {0, 1};
        data.output = new double[] {1};
        truthTable.add(data);

        data = new TraningData();
        data.intput = new double[] {1, 0};
        data.output = new double[] {1};
        truthTable.add(data);

        data = new TraningData();
        data.intput = new double[] {1, 1};
        data.output = new double[] {0};
        truthTable.add(data);


    }


    @Test
    public void testXOR() {

        NeuralNet neuralNet = new NeuralNetworkBuilder()
                .withInputs(2, LinearNuron.class)
                .addHiddenLayer(2, SigmoidNuron.class)
                .withOutputs(1, LinearNuron.class)
                .build();

        MiniBatchCreator miniBatchCreator = new MiniBatchCreator() {
            @Override
            public Collection<TraningData> createMiniBatch() {
                return truthTable;
            }
        };

        Backpropagation backpropagation = new Backpropagation();
        backpropagation.neuralNet = neuralNet;
        backpropagation.learningRate = 0.1;
        backpropagation.miniBatchCreator = miniBatchCreator;
        backpropagation.train(0.0, 10000);

        double output = neuralNet.setInputs(new double[] {0, 0}).getOutputs()[0];
        assertEquals(0, output, 0.1);

        output = neuralNet.setInputs(new double[]{0, 1}).getOutputs()[0];
        assertEquals(1, output, 0.1);

        output = neuralNet.setInputs(new double[]{1, 0}).getOutputs()[0];
        assertEquals(1, output, 0.1);

        output = neuralNet.setInputs(new double[]{1, 1}).getOutputs()[0];
        assertEquals(0, output, 0.1);

        System.out.println("done");

    }
}
