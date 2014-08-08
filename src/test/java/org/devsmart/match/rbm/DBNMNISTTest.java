package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class DBNMNISTTest {

    Random r = new Random(1);

    @Test
    public void testTrainMNISTDBN() throws Exception {
        final MNISTImageFile imageFile = new MNISTImageFile(new File("build/data/train-images-idx3-ubyte"));

        DBN dbn = new DBNBuilder()
                .addLayer(imageFile.width*imageFile.height, false)
                .addLayer(25, false)
                //.addLayer(500, false)
                //.addLayer(2000, false)
                .withRandom(r)
                .build();

        RBM rbm = dbn.rbms.get(0);

        //final int numSamples = imageFile.numImages;
        final int numSamples = 50;
        final ArrayList<Integer> images = new ArrayList<Integer>(numSamples);
        for(int i=0;i<numSamples;i++){
            images.add(i);
        }
        final MiniBatchCreator miniBatchCreator = new MiniBatchCreator() {

            final int batchSize = 1;

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
                    //RealVector reconstruct = dbn.propagateDown(dbn.propagateUp(visible, dbn.rbms.size(), r), dbn.rbms.size(), r);
                    RealVector reconstruct = dbn.rbms.get(0).getVisibleInput(dbn.rbms.get(0).activateHidden(visible, r));
                    BufferedImage image = imageFile.createImage(reconstruct.toArray());
                    File outputFile = new File(String.format("testr%d.png", i));
                    ImageIO.write(image, "png", outputFile);
                }
            }

            for(int i=0;i<5;i++){
                RealVector hidden = new ArrayRealVector(dbn.rbms.get(0).hidden.length);
                hidden.setEntry(i%dbn.rbms.get(0).hidden.length, 1);

                //RealVector reconstruct = dbn.propagateDown(hidden, dbn.rbms.size(), r);
                RealVector reconstruct = dbn.rbms.get(0).getVisibleInput(hidden);
                BufferedImage image = imageFile.createImage(reconstruct.toArray());
                File outputFile = new File(String.format("n%d.png", i));
                ImageIO.write(image, "png", outputFile);
            }

        }



    }
}
