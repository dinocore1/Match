package org.devsmart.match.rbm;


import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class DBNMNISTTest {

    @Test
    public void testTrainMNISTDBN() throws Exception {
        final MNISTImageFile imageFile = new MNISTImageFile(new File("build/data/train-images-idx3-ubyte"));

        DBN dbn = new DBNBuilder().addLayer(imageFile.width*imageFile.height, true)
                .addLayer(500, false)
                .addLayer(500, false)
                .addLayer(2000, false)
                .withRandom(new Random(1))
                .build();

        final ArrayList<Integer> images = new ArrayList<Integer>(imageFile.numImages);
        for(int i=0;i<imageFile.numImages;i++){
            images.add(i);
        }
        DBNTrainer trainer = new DBNTrainer(dbn, new DBNTrainer.Callback() {

            Random r = new Random(1);
            final int batchSize = 30;

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
        });

        trainer.train();



    }
}