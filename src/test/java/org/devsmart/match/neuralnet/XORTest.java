package org.devsmart.match.neuralnet;


import org.devsmart.match.rbm.nuron.SigmoidNuron;
import org.junit.Test;

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
                .withInputs(2, SigmoidNuron.class)
                .addHiddenLayer(2, SigmoidNuron.class)
                .withOutputs(1, SigmoidNuron.class)
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
        backpropagation.train(0.0, 1000);


    }
}
