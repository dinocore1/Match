package org.devsmart.match.neuralnet;


import org.junit.Test;
import static org.junit.Assert.*;

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
    public void upTest() {
        RBM2 rbm = new RBM2(4, 2);

        float[] weights = new float[rbm.numHidden*rbm.numVisible+rbm.numHidden+rbm.numVisible];
        for(int i=0;i<rbm.numHidden*rbm.numVisible;i++){
            weights[i] = -4;
        }
        weights[rbm.getWeight(0, 1)] = 10000;
        rbm.setWeights(weights);

        float[] hidden = rbm.activateHidden(new float[]{1, 1, 1, 1});

        assertEquals(0, hidden[0], 0.01);
        assertEquals(1.0, hidden[1], 0.01);

    }

    @Test
    public void downTest() {
        RBM2 rbm = new RBM2(4, 2);

        float[] weights = new float[rbm.numHidden*rbm.numVisible+rbm.numHidden+rbm.numVisible];
        for(int i=0;i<rbm.numHidden*rbm.numVisible;i++){
            weights[i] = -6;
        }
        weights[rbm.getWeight(0, 1)] = 10000;
        rbm.setWeights(weights);

        float[] visible = rbm.activateVisible(new float[]{0, 1});

        assertEquals(1.0, visible[0], 0.01);
        assertEquals(0, visible[1], 0.01);
        assertEquals(0, visible[2], 0.01);
        assertEquals(0, visible[3], 0.01);
    }

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

        trainer.train(4000);


        float[] hidden = rbm.activateHidden(new float[]{
                0, 1, 1,
                0, 0, 0,
                0, 0, 0});
        float[] visible = rbm.activateVisible(hidden);
        assertEquals(1, visible[0], 0.1);
        assertEquals(1, visible[1], 0.1);
        assertEquals(1, visible[2], 0.1);
        assertEquals(0, visible[3], 0.1);
        assertEquals(0, visible[4], 0.1);
        assertEquals(0, visible[5], 0.1);
        assertEquals(0, visible[6], 0.1);
        assertEquals(0, visible[7], 0.1);
        assertEquals(0, visible[8], 0.1);

        hidden = rbm.activateHidden(new double[]{
                0, 0, 0,
                0, 1, 0,
                1, 0, 0});
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible[0], 0.1);
        assertEquals(0, visible[1], 0.1);
        assertEquals(1, visible[2], 0.1);
        assertEquals(0, visible[3], 0.1);
        assertEquals(1, visible[4], 0.1);
        assertEquals(0, visible[5], 0.1);
        assertEquals(1, visible[6], 0.1);
        assertEquals(0, visible[7], 0.1);
        assertEquals(0, visible[8], 0.1);

        hidden = rbm.activateHidden(new double[]{
                0, 0, 0,
                0, 0, 0,
                1, 0, 1});
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible[0], 0.1);
        assertEquals(0, visible[1], 0.1);
        assertEquals(0, visible[2], 0.1);
        assertEquals(0, visible[3], 0.1);
        assertEquals(0, visible[4], 0.1);
        assertEquals(0, visible[5], 0.1);
        assertEquals(1, visible[6], 0.1);
        assertEquals(1, visible[7], 0.1);
        assertEquals(1, visible[8], 0.1);

        hidden = rbm.activateHidden(new double[]{
                0, 0, 1,
                0, 0, 0,
                1, 0, 0});
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible[0], 0.1);
        assertEquals(0, visible[1], 0.1);
        assertEquals(1, visible[2], 0.1);
        assertEquals(0, visible[3], 0.1);
        assertEquals(1, visible[4], 0.1);
        assertEquals(0, visible[5], 0.1);
        assertEquals(1, visible[6], 0.1);
        assertEquals(0, visible[7], 0.1);
        assertEquals(0, visible[8], 0.1);

        hidden = rbm.activateHidden(new double[]{
                1, 0, 0,
                0, 1, 0,
                1, 1, 1});
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible[0], 0.1);
        assertEquals(0, visible[1], 0.1);
        assertEquals(0, visible[2], 0.1);
        assertEquals(0, visible[3], 0.1);
        assertEquals(0, visible[4], 0.1);
        assertEquals(0, visible[5], 0.1);
        assertEquals(1, visible[6], 0.1);
        assertEquals(1, visible[7], 0.1);
        assertEquals(1, visible[8], 0.1);

        hidden = rbm.activateHidden(new double[]{
                0, 1, 0,
                0, 0, 0,
                1, 1, 1});
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible[0], 0.1);
        assertEquals(0, visible[1], 0.1);
        assertEquals(0, visible[2], 0.1);
        assertEquals(0, visible[3], 0.1);
        assertEquals(0, visible[4], 0.1);
        assertEquals(0, visible[5], 0.1);
        assertEquals(1, visible[6], 0.1);
        assertEquals(1, visible[7], 0.1);
        assertEquals(1, visible[8], 0.1);

    }

}
