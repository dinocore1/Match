package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.devsmart.match.data.MNISTImageFile;
import org.devsmart.match.rbm.nuron.BernoulliNuron;
import org.devsmart.match.rbm.nuron.Nuron;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;

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

        RBM rbm = new RBM(createBernoliiLayer(imageFile.height*imageFile.width), createBernoliiLayer(25));


        final int numSamples = 50;
        final ArrayList<Integer> images = new ArrayList<Integer>(numSamples);
        for(int i=0;i<numSamples;i++){
            images.add(i);
        }
        final MiniBatchCreator miniBatchCreator = new MiniBatchCreator() {

            final int batchSize = 10;

            @Override
            public Collection<RealVector> createMiniBatch() {
                try {
                    Collections.shuffle(images, r);
                    ArrayList<RealVector> retval = new ArrayList<RealVector>(batchSize);
                    for (int i = 0; i < batchSize; i++) {
                        retval.add(new ArrayRealVector(imageFile.getImage(images.get(i))));
                    }
                    return retval;
                } catch (IOException e) {
                    e.printStackTrace();
                    return null;
                }
            }
        };

        RBMTrainer trainer = new RBMTrainer(rbm, miniBatchCreator);
        trainer.setNumTrainingThreads(4);
        trainer.numEpic = 8000;
        trainer.learningRate = 0.1;

        trainer.train();


        {
            for(int i=0;i<5;i++) {
                RealVector visible = new ArrayRealVector(imageFile.getImage(images.get(i)));
                {
                    BufferedImage image = imageFile.createImage(visible.toArray());
                    File outputFile = new File(String.format("test%d.png", i));
                    ImageIO.write(image, "png", outputFile);
                }

                {
                    RealVector reconstruct = rbm.getVisibleInput(rbm.activateHidden(visible, r));
                    BufferedImage image = imageFile.createImage(reconstruct.toArray());
                    File outputFile = new File(String.format("testr%d.png", i));
                    ImageIO.write(image, "png", outputFile);
                }
            }

            for(int i=0;i<5;i++){
                RealVector hidden = new ArrayRealVector(rbm.hidden.length);
                hidden.setEntry(i%rbm.hidden.length, 1);

                RealVector reconstruct = rbm.getVisibleInput(hidden);
                BufferedImage image = imageFile.createImage(reconstruct.toArray());
                File outputFile = new File(String.format("n%d.png", i));
                ImageIO.write(image, "png", outputFile);
            }

        }
    }
}
