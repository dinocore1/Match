package org.devsmart.match.rbm;


import org.devsmart.match.data.MNISTImageFile;
import org.devsmart.match.rbm.nuron.BernoulliNuron;
import org.devsmart.match.rbm.nuron.Nuron;
import org.junit.Test;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;

import javax.imageio.ImageIO;

public class RBMMNISTTest {

    final Random r = new Random(1);

    private Nuron[] createBernoliiLayer(int size) {
        Nuron[] retval = new Nuron[size];
        for(int i=0;i<retval.length;i++){
            retval[i] = new BernoulliNuron();
        }
        return retval;
    }

    @Test
    public void trainMNIST() throws Exception {
        final MNISTImageFile imageFile = new MNISTImageFile(new File("build/data/train-images-idx3-ubyte"));

        RBM rbm = new RBM(createBernoliiLayer(imageFile.height*imageFile.width), createBernoliiLayer(100));


        //final int numSamples = 50;
        final int numSamples = imageFile.numImages;
        final ArrayList<Integer> images = new ArrayList<Integer>(numSamples);
        for(int i=0;i<numSamples;i++){
            images.add(i);
        }
        final RBMMiniBatchCreator miniBatchCreator = new RBMMiniBatchCreator() {

            final int batchSize = 100;

            @Override
            public Collection<double[]> createMiniBatch() {
                try {
                    Collections.shuffle(images, r);
                    ArrayList<double[]> retval = new ArrayList<double[]>(batchSize);
                    for (int i = 0; i < batchSize; i++) {
                        double[] rawPixData = imageFile.getImage(images.get(i));
                        for(int q=0;q<rawPixData.length;q++){
                            rawPixData[q] = rawPixData[q] > 0.3 ? 1.0 : 0.0;
                        }
                        retval.add(rawPixData);
                    }
                    return retval;
                } catch (IOException e) {
                    e.printStackTrace();
                    return null;
                }
            }
        };

        RBMTrainer trainer = new RBMTrainer(rbm, miniBatchCreator);
        trainer.learningRate = 0.3;
        //trainer.lossFunction = LossFunction.CrossEntropy;
        trainer.train(1, 5000);


        {
            for(int i=0;i<5;i++) {
                double[] visible = imageFile.getImage(images.get(i));
                {
                    BufferedImage image = imageFile.createImage(visible);
                    File outputFile = new File(String.format("test%d.png", i));
                    ImageIO.write(image, "png", outputFile);
                }

                {
                    double[] reconstruct = rbm.activateVisible(rbm.activateHidden(visible));
                    BufferedImage image = imageFile.createImage(reconstruct);
                    File outputFile = new File(String.format("testr%d.png", i));
                    ImageIO.write(image, "png", outputFile);
                }
            }

            for(int i=0;i<5;i++){
                double[] hidden = new double[rbm.hidden.length];
                hidden[i%rbm.hidden.length] = 1;

                double[] reconstruct = rbm.getVisibleInput(hidden);
                BufferedImage image = imageFile.createImage(reconstruct);
                File outputFile = new File(String.format("n%d.png", i));
                ImageIO.write(image, "png", outputFile);
            }

        }
    }
}
