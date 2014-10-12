package org.devsmart.match.rbm;


import org.devsmart.match.rbm.nuron.SigmoidNuron;
import org.devsmart.match.rbm.nuron.Neuron;
import org.devsmart.match.rbm.nuron.SoftPlus;
import org.junit.Test;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class RBMTest {

    private Neuron[] createBernoliiLayer(int size) {
        Neuron[] retval = new Neuron[size];
        for(int i=0;i<retval.length;i++){
            retval[i] = new SigmoidNuron();
        }
        return retval;
    }

    private Neuron[] createSoftPlusLayer(int size) {
        Neuron[] retval = new Neuron[size];
        for(int i=0;i<retval.length;i++){
            retval[i] = new SoftPlus();
        }
        return retval;
    }

    @Test
    public void testSave() throws Exception {
        RBM rbm = new RBM(createBernoliiLayer(5), createSoftPlusLayer(3));

        File saveFile = File.createTempFile("test", "rbm");
        FileOutputStream fout = new FileOutputStream(saveFile);
        rbm.save(fout);
        fout.close();

        DataInputStream din = new DataInputStream(new FileInputStream(saveFile));
        RBM rbm2 = RBM.load(din);
        din.close();

        assertEquals(rbm.W.getRowDimension(), rbm2.W.getRowDimension());
        assertEquals(rbm.W.getColumnDimension(), rbm2.W.getColumnDimension());
        for(int i=0;i<rbm.W.getRowDimension();i++){
            assertArrayEquals(rbm.W.getRow(i), rbm2.W.getRow(i), 0.0);
        }

    }

    @Test
    public void testTrainRBM() throws Exception {
        Random r = new Random(1);

        RBM rbm = new RBM(createBernoliiLayer(9), createBernoliiLayer(3));

        final ArrayList<double[]> trainingData = new ArrayList<double[]>();
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

        RBMMiniBatchCreator miniBatchCreator = new RBMMiniBatchCreator() {
            @Override
            public Collection<double[]> createMiniBatch() {
                Collections.shuffle(trainingData);
                return trainingData;
            }
        };

        RBMTrainer trainer = new RBMTrainer(rbm, miniBatchCreator);
        trainer.random = r;
        trainer.weightDecayCoefficient = 0.00001;
        trainer.train(-1.0, 10, 0.01, 5000);

        double[] hidden = rbm.activateHidden(new double[]{
                0, 1, 1,
                0, 0, 0,
                0, 0, 0});
        double[] visible = rbm.activateVisible(hidden);
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
