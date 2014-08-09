package org.devsmart.match.rbm;


import org.devsmart.match.data.MNISTImageFile;
import org.devsmart.match.data.MNISTLabelFile;
import org.junit.Test;

import java.io.File;
import java.util.*;

public class MNISTClassifierTest {

    Random r = new Random(1);

    @Test
    public void testTrainMNISTDBN() throws Exception {
        final MNISTImageFile imageFile = new MNISTImageFile(new File("build/data/train-images-idx3-ubyte"));
        final MNISTLabelFile labelFile = new MNISTLabelFile(new File("build/data/train-labels-idx1-ubyte"));


        DBNBuilder builder = new DBNBuilder();
        builder.addBernouliiLayer(imageFile.height*imageFile.width);
        builder.addBernouliiLayer(25);
        builder.addBernouliiLayer(25, 10);
        builder.addBernouliiLayer(2000);

        DBN dbn = builder.build();






    }
}
