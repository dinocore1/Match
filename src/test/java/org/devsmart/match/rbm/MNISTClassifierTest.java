package org.devsmart.match.rbm;


import org.devsmart.match.data.MNISTImageFile;
import org.devsmart.match.data.MNISTLabelFile;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class MNISTClassifierTest {

    Random r = new Random(1);

    @Test
    public void testTrainMNISTDBN() throws Exception {
        final MNISTImageFile imageFile = new MNISTImageFile(new File("build/data/train-images-idx3-ubyte"));
        final MNISTLabelFile labelFile = new MNISTLabelFile(new File("build/data/train-labels-idx1-ubyte"));

        DBNBuilder builder = new DBNBuilder();
        builder.addBernouliiLayer(imageFile.height*imageFile.width);
        builder.addBernouliiLayer(100);
        builder.addBernouliiLayer(100, 10);
        builder.addBernouliiLayer(2000);

        final DBN dbn = builder.build();

        //final int numSamples = 50;
        final int numSamples = imageFile.numImages;
        final ArrayList<Integer> images = new ArrayList<Integer>(numSamples);
        for(int i=0;i<numSamples;i++){
            images.add(i);
        }
        DBNMiniBatchCreator miniBatchCreator = new DBNMiniBatchCreator() {

            final int batchSize = 10;

            @Override
            public Collection<double[][][]> createMinibatch() {
                try {
                    Collections.shuffle(images, r);
                    ArrayList<double[][][]> retval = new ArrayList<double[][][]>(batchSize);
                    for (int i = 0; i < batchSize; i++) {
                        double[][][] input = new double[dbn.mLayers.size()][2][];
                        input[0][0] = imageFile.getImage(images.get(i));
                        input[2][1] = new double[10];
                        input[2][1][labelFile.getLabel(images.get(i))] = 1;
                        retval.add(input);
                    }
                    return retval;
                } catch (IOException e) {
                    e.printStackTrace();
                    return null;
                }
            }
        };


        DBNTrainer trainer = new DBNTrainer(dbn, miniBatchCreator);
        trainer.numEpocsPerLayer = 2000;
        trainer.learningRate = 0.2;
        trainer.numGibbsSteps = 3;
        trainer.train();

        int numCorrect = 0;
        int total = 0;
        for(int i=0;i<500;i++){
            final int knownDigit = labelFile.getLabel(i);
            double[] bottomVisible = imageFile.getImage(i);
            System.out.println(String.format("MNIST# %d Digit %d:", i, knownDigit));

            double lowestFreeEnergy = Double.NaN;
            int bestDigitClass = 0;
            for(int digit=0;digit<10;digit++) {
                double[][][] input = new double[dbn.mLayers.size()][2][];
                input[0][0] = bottomVisible;
                input[2][1] = new double[10];
                input[2][1][digit] = 1;

                double[] visible = dbn.propagateUp(2, input, r);
                visible = dbn.getVisibleForLayer(2, input, visible);
                //visible = dbn.mRBMs.get(2).getVisibleInput(visible);

                double freeEnergy = dbn.mRBMs.get(2).freeEnergy(visible);

                System.out.println(String.format("digit: %d free energy: %.5g", digit, freeEnergy));

                if(Double.isNaN(lowestFreeEnergy) || freeEnergy < lowestFreeEnergy){
                    bestDigitClass = digit;
                    lowestFreeEnergy = freeEnergy;
                }
            }

            final boolean isCorrect = knownDigit == bestDigitClass;
            System.out.println(String.format("chosen class: %d correct: %s", bestDigitClass, isCorrect));
            if(isCorrect){
                numCorrect++;
            }
            total++;
        }

        System.out.println(String.format("Accuracy: %d/%d %.5g%%", numCorrect, total, 100 * ((double)numCorrect / total)));

    }
}
