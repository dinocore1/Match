package org.devsmart.match.data;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

public class MNISTExport {

    public static void main(String[] args) {

        try {
            int imageNum = Integer.parseInt(args[0]);
            final MNISTImageFile imageFile = new MNISTImageFile(new File("build/data/train-images-idx3-ubyte"));

            BufferedImage image = imageFile.getBufferedImage(imageNum);
            File outputFile = new File("test.png");
            ImageIO.write(image, "png", outputFile);

        } catch (Exception e){
            e.printStackTrace();
        }

    }
}
