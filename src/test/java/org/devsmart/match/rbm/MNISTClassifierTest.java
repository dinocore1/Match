package org.devsmart.match.rbm;


import org.devsmart.match.data.MNISTImageFile;
import org.devsmart.match.data.MNISTLabelFile;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;

public class MNISTClassifierTest {

    Random r = new Random(1);

    @Test
    public void testTrainMNISTDBN() throws Exception {
        final MNISTImageFile imageFile = new MNISTImageFile(new File("build/data/train-images-idx3-ubyte"));
        final MNISTLabelFile labelFile = new MNISTLabelFile(new File("build/data/train-labels-idx1-ubyte"));

        DBNBuilder builder = new DBNBuilder();
        builder.addBernouliiLayer(imageFile.height*imageFile.width);
        //builder.addBernouliiLayer(100);
        builder.addBernouliiLayer(100, 10);
        builder.addBernouliiLayer(500);

        final DBN dbn = builder.build();

        //final int numSamples = 50;
        final int numSamples = imageFile.numImages;
        final ArrayList<Integer> images = new ArrayList<Integer>(numSamples);
        for(int i=0;i<numSamples;i++){
            images.add(i);
        }
        DBNMiniBatchCreator miniBatchCreator = new DBNMiniBatchCreator() {

            final int batchSize = 20;

            @Override
            public Collection<double[][]> createMinibatch() {
                try {
                    Collections.shuffle(images, r);
                    ArrayList<double[][]> retval = new ArrayList<double[][]>(batchSize);
                    for (int i = 0; i < batchSize; i++) {
                        double[][] input = new double[2][];

                        double[] rawPixData = imageFile.getImage(images.get(i));
                        for(int q=0;q<rawPixData.length;q++){
                            rawPixData[q] = rawPixData[q] > 0.3 ? 1.0 : 0.0;
                        }

                        input[0] = rawPixData;
                        input[1] = new double[10];
                        input[1][labelFile.getLabel(images.get(i))] = 1;
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
        //trainer.lossFunction = LossFunction.CrossEntropy;
        trainer.learningRate = 0.1;
        trainer.numGibbsSteps = 1;
        trainer.momentum = 0.9;
        trainer.weightDecayCoefficient = 0.001;
        trainer.train(-1, 2000);

        int numCorrect = 0;
        int total = 0;
        for(int i=0;i<500;i++){
            final int knownDigit = labelFile.getLabel(i);
            double[] bottomVisible = imageFile.getImage(i);
            for(int q=0;q<bottomVisible.length;q++){
                bottomVisible[q] = bottomVisible[q] > 0.3 ? 1.0 : 0.0;
            }
            System.out.println(String.format("MNIST# %d Digit %d:", i, knownDigit));

            double lowestFreeEnergy = Double.NaN;
            int bestDigitClass = 0;
            for(int digit=0;digit<10;digit++) {
                double[][] input = new double[2][];
                input[0] = bottomVisible;
                input[1] = new double[10];
                input[1][digit] = 1;

                double[] visible = dbn.propagateUp(1, input, r);
                double freeEnergy = dbn.mRBMs.get(1).freeEnergy(visible);

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
