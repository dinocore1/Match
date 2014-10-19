package org.devsmart.match.neuralnet;


import org.junit.Test;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;

public class RBM2Test {

    static ArrayList<double[]> trainingData;

    static {
        trainingData = new ArrayList<double[]>();
        trainingData.add(new double[]{
                0, 0, 0,
                0, 0, 0,
                1, 1, 1});
        trainingData.add(new double[]{
                0, 0, 1,
                0, 1, 0,
                1, 0, 0});
        trainingData.add(new double[]{
                1, 1, 1,
                0, 0, 0,
                0, 0, 0});
    }

    static Random r = new Random(1);

    @Test
    public void testRBM2() {
        RBM2 rbm = new RBM2(9, 3);

        RBMTrainer2 trainer = new RBMTrainer2(rbm, new MiniBatchCreator() {
            @Override
            public Collection<TraningData> createMiniBatch() {
                ArrayList<TraningData> retval = new ArrayList<TraningData>(3);
                for(double[] input : trainingData) {
                    retval.add(new TraningData(input));
                }

                Collections.shuffle(retval, r);
                return retval;
            }
        });

        trainer.train(1000);

    }
}
