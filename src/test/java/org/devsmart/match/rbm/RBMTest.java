package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.devsmart.match.rbm.RBM;
import org.devsmart.match.rbm.RBMTrainer;
import org.devsmart.match.rbm.nuron.BernoulliNuron;
import org.devsmart.match.rbm.nuron.Nuron;
import org.junit.Test;
import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;

public class RBMTest {

    private Nuron[] createBernoliiLayer(int size) {
        Nuron[] retval = new Nuron[size];
        for(int i=0;i<retval.length;i++){
            retval[i] = new BernoulliNuron();
        }
        return retval;
    }

    @Test
    public void testTrainRBM() throws Exception {
        Random r = new Random(1);

        RBM rbm = new RBM(createBernoliiLayer(9), createBernoliiLayer(3));

        final ArrayList<RealVector> trainingData = new ArrayList<RealVector>();
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

        MiniBatchCreator miniBatchCreator = new MiniBatchCreator() {
            @Override
            public Collection<RealVector> createMiniBatch() {
                Collections.shuffle(trainingData);
                return trainingData;
            }
        };

        RBMTrainer trainer = new RBMTrainer(rbm, miniBatchCreator);
        trainer.random = r;
        //trainer.setNumTrainingThreads(4);
        trainer.numEpic = 2000;
        trainer.train();

        RealVector hidden = rbm.activateHidden(new ArrayRealVector(new double[]{
                0, 1, 1,
                0, 0, 0,
                0, 0, 0}), r);
        RealVector visible = rbm.getVisibleInput(hidden);
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
                1, 0, 0}), r);
        visible = rbm.getVisibleInput(hidden);
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
                1, 0, 1}), r);
        visible = rbm.getVisibleInput(hidden);
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
                1, 0, 0}),r );
        visible = rbm.getVisibleInput(hidden);
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
                0, 1, 0,
                1, 1, 1}), r);
        visible = rbm.getVisibleInput(hidden);
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
                0, 1, 0,
                0, 0, 0,
                1, 1, 1}), r);
        visible = rbm.getVisibleInput(hidden);
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
