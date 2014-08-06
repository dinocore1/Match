package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.devsmart.match.rbm.RBM;
import org.devsmart.match.rbm.RBMTrainer;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Random;

public class RBMTest {

    @Test
    public void testTrainRBM() {

        RBM rbm = new RBM(9, 3, false, new Random(1));

        ArrayList<RealVector> trainingData = new ArrayList<RealVector>();
        trainingData.add(new ArrayRealVector(new double[]{
                0, 0, 0,
                0, 0, 0,
                1, 1, 1}));
        trainingData.add(new ArrayRealVector(new double[]{
                0, 0, 1,
                0, 1, 0,
                1, 0, 0}));
        trainingData.add(new ArrayRealVector(new double[]{
                1, 1, 1,
                0, 0, 0,
                0, 0, 0}));

        RBMTrainer trainer = new RBMTrainer(rbm);
        //trainer.train(trainingData, 10000);

        RealVector hidden = rbm.activateHidden(new ArrayRealVector(new double[]{
                0, 1, 1,
                0, 0, 0,
                0, 0, 0}));
        RealVector visible = rbm.activateVisible(hidden);
        assertEquals(1, visible.getEntry(0), 0.1);
        assertEquals(1, visible.getEntry(1), 0.1);
        assertEquals(1, visible.getEntry(2), 0.1);
        assertEquals(0, visible.getEntry(3), 0.1);
        assertEquals(0, visible.getEntry(4), 0.1);
        assertEquals(0, visible.getEntry(5), 0.1);
        assertEquals(0, visible.getEntry(6), 0.1);
        assertEquals(0, visible.getEntry(7), 0.1);
        assertEquals(0, visible.getEntry(8), 0.1);

        hidden = rbm.activateHidden(new ArrayRealVector(new double[]{
                0, 0, 0,
                0, 1, 0,
                1, 0, 0}));
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible.getEntry(0), 0.1);
        assertEquals(0, visible.getEntry(1), 0.1);
        assertEquals(1, visible.getEntry(2), 0.1);
        assertEquals(0, visible.getEntry(3), 0.1);
        assertEquals(1, visible.getEntry(4), 0.1);
        assertEquals(0, visible.getEntry(5), 0.1);
        assertEquals(1, visible.getEntry(6), 0.1);
        assertEquals(0, visible.getEntry(7), 0.1);
        assertEquals(0, visible.getEntry(8), 0.1);

        hidden = rbm.activateHidden(new ArrayRealVector(new double[]{
                0, 0, 0,
                0, 0, 0,
                1, 0, 1}));
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible.getEntry(0), 0.1);
        assertEquals(0, visible.getEntry(1), 0.1);
        assertEquals(0, visible.getEntry(2), 0.1);
        assertEquals(0, visible.getEntry(3), 0.1);
        assertEquals(0, visible.getEntry(4), 0.1);
        assertEquals(0, visible.getEntry(5), 0.1);
        assertEquals(1, visible.getEntry(6), 0.1);
        assertEquals(1, visible.getEntry(7), 0.1);
        assertEquals(1, visible.getEntry(8), 0.1);

        hidden = rbm.activateHidden(new ArrayRealVector(new double[]{
                0, 0, 1,
                0, 0, 0,
                1, 0, 0}));
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible.getEntry(0), 0.1);
        assertEquals(0, visible.getEntry(1), 0.1);
        assertEquals(1, visible.getEntry(2), 0.1);
        assertEquals(0, visible.getEntry(3), 0.1);
        assertEquals(1, visible.getEntry(4), 0.1);
        assertEquals(0, visible.getEntry(5), 0.1);
        assertEquals(1, visible.getEntry(6), 0.1);
        assertEquals(0, visible.getEntry(7), 0.1);
        assertEquals(0, visible.getEntry(8), 0.1);

        hidden = rbm.activateHidden(new ArrayRealVector(new double[]{
                1, 0, 0,
                0, 0, 0,
                1, 0, 1}));
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible.getEntry(0), 0.1);
        assertEquals(0, visible.getEntry(1), 0.1);
        assertEquals(0, visible.getEntry(2), 0.1);
        assertEquals(0, visible.getEntry(3), 0.1);
        assertEquals(0, visible.getEntry(4), 0.1);
        assertEquals(0, visible.getEntry(5), 0.1);
        assertEquals(1, visible.getEntry(6), 0.1);
        assertEquals(1, visible.getEntry(7), 0.1);
        assertEquals(1, visible.getEntry(8), 0.1);

        hidden = rbm.activateHidden(new ArrayRealVector(new double[]{
                1, 1, 0,
                0, 0, 0,
                1, 1, 1}));
        visible = rbm.activateVisible(hidden);
        assertEquals(0, visible.getEntry(0), 0.1);
        assertEquals(0, visible.getEntry(1), 0.1);
        assertEquals(0, visible.getEntry(2), 0.1);
        assertEquals(0, visible.getEntry(3), 0.1);
        assertEquals(0, visible.getEntry(4), 0.1);
        assertEquals(0, visible.getEntry(5), 0.1);
        assertEquals(1, visible.getEntry(6), 0.1);
        assertEquals(1, visible.getEntry(7), 0.1);
        assertEquals(1, visible.getEntry(8), 0.1);

    }
}
